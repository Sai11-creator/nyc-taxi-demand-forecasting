"""Read-only audit of saved taxi panels, features, predictions and bootstrap.
No model is fitted, and no input parquet or notebook is written.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/taxi-matplotlib')
import numpy as np,pandas as pd,geopandas as gpd
from arch.bootstrap import MovingBlockBootstrap


def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--check-raw',action='store_true');a=p.parse_args();r=a.project;out=a.output;out.mkdir(parents=True,exist_ok=True)
 keys=['tpep_pickup_datetime','PULocationID'];data={};monthly=[];checks=[];annual=[]
 def check(label,condition):
  if not condition:raise AssertionError(label)
  checks.append(label)
 for year in [2024,2025]:
  agg=pd.read_parquet(r/f'data/yellow_taxi_{year}_aggreg.parquet');base=pd.read_parquet(r/f'data/yellow_taxi_{year}_features.parquet');sp=pd.read_parquet(r/f'data/yellow_taxi_{year}_features_neighbor_mean.parquet');shock=pd.read_parquet(r/f'data/yellow_taxi_{year}_features_neighbor_shock.parquet')
  hours=pd.date_range(str(year),str(year+1),freq='h',inclusive='left');expected=pd.MultiIndex.from_product([hours,range(1,264)],names=keys)
  check(f'{year}: complete ordered hourly grid and unique zone-hours',pd.MultiIndex.from_frame(agg[keys]).equals(expected))
  check(f'{year}: nonnegative integer targets',agg.trip_count.notna().all() and agg.trip_count.ge(0).all() and pd.api.types.is_integer_dtype(agg.trip_count))
  for label,frame in [('base',base),('spatial',sp),('shock',shock)]:
   pd.testing.assert_frame_equal(frame[keys+['trip_count']],agg,check_dtype=False);checks.append(f'{year}: {label} keys/targets match aggregate exactly')
  pd.testing.assert_frame_equal(base,sp[base.columns]);pd.testing.assert_frame_equal(base,shock[base.columns]);checks.append(f'{year}: all temporal feature columns identical between variants')
  data[year]={'agg':agg,'base':base,'sp':sp,'shock':shock}
  annual.append(dict(year=year,rows=len(agg),hours=len(hours),pickups=int(agg.trip_count.sum()),mean=float(agg.trip_count.mean()),zero_fraction=float(agg.trip_count.eq(0).mean())))
  for month,g in agg.groupby(agg[keys[0]].dt.month):monthly.append(dict(year=year,month=int(month),zone_hours=len(g),pickups=int(g.trip_count.sum()),zero_fraction=float(g.trip_count.eq(0).mean()),mean=float(g.trip_count.mean())))
 pd.DataFrame(monthly).to_csv(out/'panel_monthly_summary.csv',index=False)
 (out/'panel_annual_summary.json').write_text(json.dumps(annual,indent=2))
 validation=data[2024]['base'].loc[data[2024]['base'].tpep_pickup_datetime.dt.month==9];target=validation.trip_count.to_numpy();baselines=[]
 for name,col in [('Previous hour','lag_1h'),('Previous day','lag_24h'),('Previous week','lag_1w'),('Zero',None)]:
  prediction=validation[col].to_numpy() if col else np.zeros(len(validation));error=prediction-target;baselines.append({'predictor':name,'mae':float(np.abs(error).mean()),'rmse':float(np.sqrt(np.square(error).mean()))})
 pd.DataFrame(baselines).to_csv(out/'development_baselines.csv',index=False)
 print('Panel and cross-variant identity checks passed',flush=True)
 counts=pd.concat([data[y]['agg'] for y in [2024,2025]]).pivot(index=keys[0],columns=keys[1],values='trip_count').sort_index()
 for year in [2024,2025]:
  frame=data[year]['base'];hours=pd.DatetimeIndex(frame[keys[0]].unique());row=counts.index.get_indexer(frame[keys[0]]);col=counts.columns.get_indexer(frame[keys[1]])
  for name,lag in [('lag_1h',1),('lag_24h',24),('lag_1w',168)]:check(f'{year}: exact same-zone {name}',np.allclose(frame[name],counts.shift(lag).to_numpy()[row,col],equal_nan=True))
  for window in [3,6,24]:check(f'{year}: past-only rolling {window}',np.allclose(frame[f'roll_mean_{window}'],counts.shift(1).rolling(window).mean().to_numpy()[row,col],equal_nan=True))
  for name,values in [('hours',frame[keys[0]].dt.hour),('day_of_the_week',frame[keys[0]].dt.dayofweek),('month',frame[keys[0]].dt.month),('is_weekend',(frame[keys[0]].dt.dayofweek>=5).astype(int))]:check(f'{year}: calendar {name}',np.array_equal(frame[name],values))
 zones=gpd.read_file(r/'taxi_zones/taxi_zones.shp');check('Shapefile positional order agrees with LocationID 1--263',np.array_equal(zones.LocationID,np.arange(1,264)))
 zones=zones.set_index('LocationID').sort_index();ids=zones.index;adj=np.zeros((263,263));edges=[]
 for z in ids:
  shared=zones.boundary.intersection(zones.loc[z].geometry.boundary)
  for j in ids[(~shared.is_empty)&(ids!=z)]:adj[z-1,j-1]=1;edges.append(dict(zone=int(z),neighbor=int(j),shared_length=float(shared.loc[j].length),point_only=bool(shared.loc[j].length==0)))
 check('Adjacency is symmetric without self-edges',np.array_equal(adj,adj.T) and not adj.diagonal().any());neighbors=adj.sum(1);has=neighbors>0;isolated=ids[~has].tolist();check('Six isolated zones',isolated==[1,46,103,104,105,202]);pd.DataFrame(edges).to_csv(out/'adjacency_edges.csv',index=False)
 train_counts=counts.loc[counts.index<pd.Timestamp('2024-09-01')];zone_hour=train_counts.groupby(train_counts.index.hour).mean()
 c24=counts.loc['2024'];past=c24.groupby(c24.index.hour).transform(lambda x:x.expanding().mean().shift(1))
 for year in [2024,2025]:
  frame=data[year]['sp'];hours=pd.DatetimeIndex(frame[keys[0]].unique());prev=hours-pd.Timedelta(hours=1);v=counts.reindex(prev).to_numpy();signal=v@adj.T;signal[:,has]/=neighbors[has];signal[:,~has]=0
  row=hours.get_indexer(frame[keys[0]]);col=ids.get_indexer(frame[keys[1]]);check(f'{year}: neighbor mean matches exact t-1 adjacency',np.allclose(frame.neighbors_mean_PU,signal[row,col],equal_nan=True))
  baseline=past.reindex(prev).to_numpy(copy=True) if year==2024 else np.empty_like(v,dtype=float)
  evaluation=hours>=pd.Timestamp('2024-09-01');baseline[evaluation]=zone_hour.reindex(prev[evaluation].hour).to_numpy();sh=v-baseline;signal=np.nan_to_num(sh,nan=0)@adj.T;signal[:,has]/=neighbors[has];missing=np.isnan(sh).astype(float)@adj.T>0;signal[missing]=np.nan;signal[:,~has]=0
  check(f'{year}: shock matches expanding-training/frozen-evaluation formula',np.allclose(data[year]['shock'].neighbors_mean_shock_1h,signal[row,col],equal_nan=True))
 print('All saved temporal and spatial features matched reconstructed formulas',flush=True)
 means=data[2024]['base'].groupby(keys[1]).trip_count.mean();means8=train_counts.mean();rank=means.sort_values(ascending=False).index.tolist();groups={'High':rank[:87],'Medium':rank[87:174],'Low':rank[174:]}
 (out/'demand_group_membership.json').write_text(json.dumps({k:list(map(int,v)) for k,v in groups.items()},indent=2))
 category={}
 for label,mu in [('Full2024',means),('JanAug2024',means8)]:
  neighbor_mean=adj@mu.reindex(ids).to_numpy();neighbor_mean[has]/=neighbors[has];neighbor_mean[~has]=np.nan
  busy=ids[(mu.reindex(ids).to_numpy()<1)&(neighbor_mean>=10)].tolist();category[label]={'seven_zones':busy,'quiet_count':int(((mu.reindex(ids).to_numpy()<1)&(neighbor_mean<10)).sum()),'higher_count':int((mu>=1).sum()),'isolated':isolated};check(label+': seven-zone rule matches',busy==[2,53,124,194,199,203,219])
 (out/'zone_categories.json').write_text(json.dumps(category,indent=2))
 pred=pd.read_parquet(r/'data/predictions_2025.parquet');frame=data[2025]['base'];check('Saved predictions have matching row count and no NaNs',len(pred)==len(frame) and pred.notna().all().all())
 y=frame.trip_count.to_numpy();errors={name:pred[col].to_numpy(dtype=float)-y for name,col in [('Basic','pred_basic'),('Spatial','pred_mean'),('Shock','pred_shock')]}
 full=[]
 for name,e in errors.items():full.append(dict(model=name,mae=float(np.abs(e).mean()),mse=float(np.square(e).mean()),rmse=float(np.sqrt(np.square(e).mean())),negative_predictions=int((pred[{'Basic':'pred_basic','Spatial':'pred_mean','Shock':'pred_shock'}[name]]<0).sum())))
 pd.DataFrame(full).to_csv(out/'annual_saved_prediction_metrics.csv',index=False)
 rows=[]
 for month in range(1,13):
  mask=frame[keys[0]].dt.month.eq(month).to_numpy();row={'month':month}
  for name,e in errors.items():row[name+'_mae']=np.abs(e[mask]).mean();row[name+'_rmse']=np.sqrt(np.square(e[mask]).mean())
  row['rmse_improvement_pct']=100*(row['Basic_rmse']-row['Spatial_rmse'])/row['Basic_rmse'];row['mae_improvement_pct']=100*(row['Basic_mae']-row['Spatial_mae'])/row['Basic_mae'];rows.append(row)
 monthly_metrics=pd.DataFrame(rows);monthly_metrics.to_csv(out/'monthly_prediction_metrics.csv',index=False);check('Spatial improves both monthly metrics all 12 months',monthly_metrics.rmse_improvement_pct.gt(0).all() and monthly_metrics.mae_improvement_pct.gt(0).all())
 loss=pd.DataFrame({'date':frame[keys[0]].dt.normalize(),'zone':frame[keys[1]],'basic_se':errors['Basic']**2,'spatial_se':errors['Spatial']**2});loss['delta']=loss.basic_se-loss.spatial_se
 zone_metrics=loss.groupby('zone')[['basic_se','spatial_se','delta']].mean();zone_metrics['relative_pct']=100*zone_metrics.delta/zone_metrics.basic_se;zone_metrics['mean2024']=means;zone_metrics['name']=zones.zone;zone_metrics.sort_values('delta',ascending=False).to_csv(out/'zone_basic_spatial_metrics.csv')
 sorted_delta=zone_metrics.delta.sort_values(ascending=False);concentration={str(k):float(sorted_delta.iloc[:k].sum()/sorted_delta.sum()) for k in [5,10]};concentration['top5_ids']=list(map(int,sorted_delta.index[:5]));concentration['top10_ids']=list(map(int,sorted_delta.index[:10]));concentration['net_sum']=float(sorted_delta.sum());(out/'concentration.json').write_text(json.dumps(concentration,indent=2))
 boot=[];regimes=[]
 for label,members in [('All',list(ids))]+list(groups.items()):
  selected=loss.loc[loss.zone.isin(members)];daily=selected.groupby('date').delta.mean().sort_index();check(label+': bootstrap has 365 calendar days',daily.index.equals(pd.date_range('2025-01-01','2025-12-31')))
  if label=='All':(out/'daily_loss_metadata.json').write_text(json.dumps({'days':len(daily),'unique_values':int(daily.nunique()),'mean':float(daily.mean())},indent=2))
  if label!='All':regimes.append(dict(group=label,zones=len(members),basic_rmse=float(np.sqrt(selected.basic_se.mean())),spatial_rmse=float(np.sqrt(selected.spatial_se.mean())),mse_improvement=float(selected.delta.mean())))
  for block in ([7,14] if label=='All' else [7,14,28]):
   reps=2000 if label=='All' else 5000;lo,hi=MovingBlockBootstrap(block,daily.to_numpy(),seed=42).conf_int(np.mean,reps=reps,method='percentile',size=.95).ravel();boot.append(dict(group=label,block_days=block,reps=reps,mean_difference=daily.mean(),lower=lo,upper=hi))
 pd.DataFrame(regimes).to_csv(out/'regime_metrics.csv',index=False);pd.DataFrame(boot).to_csv(out/'basic_spatial_bootstrap.csv',index=False)
 # Zone names, means and neighbor weights for diagnostic verification.
 diagnostics=[]
 for z in [194,74,2,199]:
  row={'zone':z,'name':zones.loc[z,'zone'],'mean2024':float(means[z]),'mean_jan_aug':float(means8[z]),'mean2025':float(frame.loc[frame[keys[1]]==z,'trip_count'].mean()),'zero2025':float(frame.loc[frame[keys[1]]==z,'trip_count'].eq(0).mean()),'neighbors':list(map(int,ids[adj[z-1]>0]))};diagnostics.append(row)
 (out/'zone_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
 raw_results=[]
 if a.check_raw:
  for year in [2024,2025]:
   agg=data[year]['agg']
   for month in range(1,13):
    path=Path.home()/'Downloads'/f'yellow_tripdata_{year}-{month:02d}.parquet';row=dict(year=year,month=month,raw_exists=path.exists())
    if path.exists():
     raw=pd.read_parquet(path,columns=keys);start=pd.Timestamp(year=year,month=month,day=1);end=start+pd.DateOffset(months=1);valid=raw[keys[1]].isin(range(1,264))&raw[keys[0]].ge(start)&raw[keys[0]].lt(end)
     retained=raw.loc[valid];g=retained.groupby([pd.Grouper(key=keys[0],freq='h'),keys[1]]).size();saved=agg.loc[agg[keys[0]].ge(start)&agg[keys[0]].lt(end)].set_index(keys).trip_count
     reconstructed=g.reindex(saved.index,fill_value=0);check(f'{year}-{month:02}: full raw-to-aggregate zone-hour equality',np.array_equal(reconstructed,saved));row.update(raw_rows=len(raw),retained=int(valid.sum()),removed=int((~valid).sum()),zone_hours_match=True)
     cleaned=r/f'data/yellow_taxi_{year}_{month:02d}_cleaned.parquet'
     if cleaned.exists():
      clean=pd.read_parquet(cleaned,columns=keys);row['cleaned_rows']=len(clean);row['cleaned_outside_month']=int((~(clean[keys[0]].ge(start)&clean[keys[0]].lt(end))).sum())
     del raw,retained
    raw_results.append(row);print('Raw audit',year,month,'exists',row['raw_exists'],flush=True)
  pd.DataFrame(raw_results).to_csv(out/'raw_monthly_verification.csv',index=False)
 (out/'checks.json').write_text(json.dumps(checks,indent=2));print('CHECKS',len(checks));print(pd.DataFrame(full));print(pd.DataFrame(regimes));print(pd.DataFrame(boot));print('CONCENTRATION',concentration)

if __name__=='__main__':main()
