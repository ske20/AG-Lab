import math
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import streamlit as st
from geostatistics_theme import setup_theme
 
st.set_page_config(
page_title="Applied Geostatistics Lab",
layout="wide"
)
 
theme_name, theme = setup_theme()
st.set_page_config(page_title='Probability Distribution Teaching App',layout='wide')
PDFS=['Normal','Truncated normal','Beta','Lognormal','Exponential','Triangular','Uniform']
LABELS=['p10','p50','p90']; COLORS=['green','darkorange','red']

def fit_all(x, names):
    mn,mx=float(x.min()),float(x.max()); span=mx-mn; out=[]
    for name in names:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if name=='Normal': d=stats.norm; par=d.fit(x); k=2
                elif name=='Truncated normal':
                    if mn<0: continue
                    d=stats.truncnorm; lo=0.; hi=mx+max(.01*span,1e-8)
                    def loss(z):
                        mu,logsd=z; sd=np.exp(logsd)
                        vals=d.logpdf(x,(lo-mu)/sd,(hi-mu)/sd,loc=mu,scale=sd)
                        return -float(vals.sum()) if np.all(np.isfinite(vals)) else 1e100
                    res=optimize.minimize(loss,[x.mean(),np.log(max(x.std(ddof=1),1e-6))],method='Nelder-Mead')
                    mu=float(res.x[0]); sd=float(np.exp(res.x[1])); par=((lo-mu)/sd,(hi-mu)/sd,mu,sd); k=2
                elif name=='Beta':
                    if span<=0: continue
                    d=stats.beta; lo=0 if mn>=0 else mn-.05*span; hi=mx+max(.05*span,1e-6)
                    par=d.fit(x,floc=lo,fscale=hi-lo); k=2
                elif name=='Lognormal':
                    if np.any(x<=0): continue
                    d=stats.lognorm; par=d.fit(x,floc=0); k=2
                elif name=='Exponential':
                    if mn<0: continue
                    d=stats.expon; par=d.fit(x,floc=0); k=1
                elif name=='Triangular': d=stats.triang; par=d.fit(x); k=3
                else: d=stats.uniform; par=(mn,span); k=2
                logpdf=d.logpdf(x,*par)
                if not np.all(np.isfinite(logpdf)): continue
                ks,pvalue=stats.kstest(x,d.cdf,args=par)
                if not np.isfinite(ks): continue
                ll=float(logpdf.sum())
                out.append(dict(name=name,dist=d,params=par,ks=ks,pvalue=pvalue,
                                aic=2*k-2*ll,bic=k*math.log(len(x))-2*ll))
        except (ValueError,OverflowError,ZeroDivisionError,FloatingPointError): pass
    return out

def make_bins(x,width):
    n=int(math.ceil(np.ptp(x)/width))
    if n>500: raise ValueError(f'This width creates {n} bins. Increase width to use at most 500 bins.')
    edges=float(x.min())+width*np.arange(n+1,dtype=float)
    if edges[-1]<=x.max(): edges=np.append(edges,edges[-1]+width)
    count,edges=np.histogram(x,bins=edges)
    return count,edges,count/len(x),np.cumsum(count)/len(x)

def bin_percentiles(count,edges):
    cumulative=np.cumsum(count); out=[]; n=int(cumulative[-1])
    for pct in (10,50,90):
        target=n*pct/100
        i=min(int(np.searchsorted(cumulative,target,side='left')),len(count)-1)
        while count[i]==0 and i<len(count)-1: i+=1
        before=cumulative[i-1] if i else 0
        out.append(edges[i]+(target-before)/count[i]*(edges[i+1]-edges[i]))
    return np.array(out)

def observed_interval(fit, minimum, maximum):
    d, par = fit['dist'], fit['params']
    lower, upper = float(d.cdf(minimum, *par)), float(d.cdf(maximum, *par))
    mass = upper - lower
    if not np.isfinite(mass) or mass <= 1e-12:
        raise ValueError(f"{fit['name']} has negligible fitted probability within the observed range.")
    return lower, upper, mass

def conditional_pdf(fit, grid, minimum, maximum):
    _, _, mass = observed_interval(fit, minimum, maximum)
    density = np.asarray(fit['dist'].pdf(grid, *fit['params']), dtype=float) / mass
    return np.where((grid >= minimum) & (grid <= maximum), density, 0.0)

def simulate(fit, n, seed, minimum, maximum):
    d, par = fit['dist'], fit['params']
    lower, upper, _ = observed_interval(fit, minimum, maximum)
    rng = np.random.default_rng(seed)
    u = lower + (upper - lower) * rng.random(n)
    u = np.clip(u, np.nextafter(lower, upper), np.nextafter(upper, lower))
    values = np.asarray(d.ppf(u, *par), dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{fit['name']} produced non-finite Monte Carlo draws.")
    return np.clip(values, minimum, maximum)

def render(fig):
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

def labels(ax,bars,kind='frequency'):
    if len(bars)>80: return
    for bar in bars:
        h=bar.get_height()
        if h>0:
            text=f'{h:.3f}' if kind=='probability' else f'{h:.0f}'
            ax.annotate(text,(bar.get_x()+bar.get_width()/2,h),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
    ax.margins(y=.15)

def percent_lines(ax,q):
    for name,val,color in zip(LABELS,q,COLORS): ax.axvline(val,color=color,ls='--',label=f'{name} = {val:.5g}')

st.title('Probability distributions and Monte Carlo')
st.caption('Observed frequencies, empirical probabilities, fitted CDF and non-negative simulation')
with st.sidebar:
    uploaded=st.file_uploader('Upload CSV',type='csv')
if uploaded is None:
    st.info('Upload a CSV file to list its columns and start analysis.'); st.stop()
try: df=pd.read_csv(uploaded)
except Exception as e: st.error(f'CSV error: {e}'); st.stop()
if df.empty: st.error('CSV contains no data rows.'); st.stop()
st.subheader('Columns in uploaded CSV'); st.write(', '.join(map(str,df.columns)))
with st.sidebar:
    col=st.selectbox('Select column',list(df.columns))
series=pd.to_numeric(df[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
x=series.dropna().to_numpy(float)
st.caption(f'Selected column: {col} | {len(x):,} valid values | {len(series)-len(x):,} missing/non-numeric values excluded')
if len(x)<5: st.error('At least five valid numeric observations are required.'); st.stop()
if np.ptp(x)==0: st.error('All values are identical; choose a column with variation.'); st.stop()
with st.sidebar:
    default=max(1,int(math.ceil(float(np.ptp(x))/12)))
    width=st.number_input('Bin width (whole units of selected column)',min_value=1,value=default,step=1,format='%d')
    chosen=st.multiselect('PDFs to fit',PDFS,default=PDFS)
    runs=st.number_input('Monte Carlo runs',min_value=100,max_value=500000,value=10000,step=100)
    seed=st.number_input('Random seed',min_value=0,max_value=2147483647,value=557)
if not chosen: st.error('Select at least one PDF.'); st.stop()
try: counts,edges,probs,cum=make_bins(x,width)
except ValueError as e: st.error(str(e)); st.stop()
with st.spinner('Fitting PDFs...'): fits=fit_all(x,chosen)
if not fits: st.error('None of the selected distributions could be fitted.'); st.stop()
best=min(fits,key=lambda f:f['ks'])['name']; names=[f['name'] for f in fits]
with st.sidebar: selected=st.selectbox('Fitted PDF for CDF and simulation',names,index=names.index(best))
fit=next(f for f in fits if f['name']==selected)
raw_q=np.percentile(x,[10,50,90]); bq=bin_percentiles(counts,edges)
st.caption(f'Selected fit: {selected} | KS statistic {fit["ks"]:.4f}. The lowest KS is suggested, not necessarily physically appropriate.')
raw,descriptive,frequency,binned,cdf,mc,gof=st.tabs(['Raw data','Descriptive statistics','Raw frequency','Binned analysis','Empirical + fitted CDF','Monte Carlo','Fit comparison'])
with raw:
    st.subheader('Raw data for selected column')
    st.dataframe(pd.DataFrame({'CSV row':series.dropna().index+2,col:x}),hide_index=True,use_container_width=True,height=500)
with descriptive:
    vals,cts=np.unique(x,return_counts=True); most=cts.max()
    mode=', '.join(f'{v:.3f}' for v in vals[cts==most][:15]) if most>1 else 'No repeated value (no informative mode)'
    if np.count_nonzero(cts==most)>15: mode+=' (more tied values)'
    q1,q2,q3=np.percentile(x,[25,50,75]); sd=x.std(ddof=1); mean=x.mean()
    rows=[('Count',len(x)),('Mean',mean),('Median',np.median(x)),('Mode(s)',mode),('Range',np.ptp(x)),('Minimum',x.min()),('Maximum',x.max()),('Q1',q1),('Q2',q2),('Q3',q3),('IQR',q3-q1),('Sample SD',sd),('CV (%)',sd/abs(mean)*100 if mean else np.nan),('Skewness',stats.skew(x,bias=False)),('Excess kurtosis',stats.kurtosis(x,fisher=True,bias=False))]
    display_rows=[(name, str(value) if name=='Count' or isinstance(value,str) else f'{value:.3f}') for name,value in rows]
    st.dataframe(pd.DataFrame(display_rows,columns=['Statistic','Value']),hide_index=True,use_container_width=True)
    st.caption('CV = sample SD / |mean| × 100; excess kurtosis of a normal distribution is 0.')
with frequency:
    st.subheader('Raw frequency: each exact distinct value, no binning')
    table=pd.DataFrame({'Value':vals,'Frequency':cts}); st.dataframe(table,hide_index=True,use_container_width=True,height=350)
    if len(vals)<=100:
        fig,ax=plt.subplots(figsize=(max(11,min(22,len(vals)*.25)),5))
        bars=ax.bar(np.arange(len(vals)),cts,color='#5B7BEA',edgecolor='black')
        ax.set_xticks(np.arange(len(vals)),[f'{v:.5g}' for v in vals],rotation=75 if len(vals)>15 else 0)
        if len(vals)<=80: labels(ax,bars)
        ax.set(xlabel=col,ylabel='Frequency',title='Raw frequency histogram'); render(fig)
    else: st.info('More than 100 unique values: full exact-frequency table is above. Use binned analysis for a readable histogram.')
with binned:
    st.subheader(f'Binned histograms (bin width {width:.6g})')
    table=pd.DataFrame({'Bin start':edges[:-1],'Bin end':edges[1:],'Frequency':counts,'Empirical probability':probs,'Cumulative probability':cum})
    st.dataframe(table,hide_index=True,use_container_width=True,height=280)
    centers=(edges[:-1]+edges[1:])/2
    fig,ax=plt.subplots(figsize=(12,5)); bars=ax.bar(centers,counts,width=np.diff(edges),color='#5B7BEA',edgecolor='black'); labels(ax,bars)
    ax.set(title='Binned frequency histogram',xlabel=col,ylabel='Frequency'); render(fig)
    fig,ax=plt.subplots(figsize=(12,5)); bars=ax.bar(centers,probs,width=np.diff(edges),color='#67A8A4',edgecolor='black'); labels(ax,bars,'probability')
    palette=plt.get_cmap('tab10')
    for i,f in enumerate(fits):
        # The probability for a bin is the CDF difference across its edges.
        expected=np.diff(f['dist'].cdf(edges,*f['params']))
        ax.plot(centers,expected,color=palette(i%10),lw=2,marker='o',ms=3,
                label=f"Fitted {f['name']} probability per bin")
    ax.set(title='Empirical probability per bin vs fitted bin probabilities',xlabel=col,ylabel='Probability per bin')
    ax.legend(fontsize=8); render(fig)
    fig,ax=plt.subplots(figsize=(12,5)); ax.step(edges,np.r_[0,cum],where='post',lw=2,label='Binned cumulative probability')
    for label,val,color,pct in zip(LABELS,bq,COLORS,[.1,.5,.9]):
        ax.axvline(val,color=color,ls='--',label=f'Binned {label} ≈ {val:.5g}'); ax.scatter([val],[pct],color=color,zorder=5)
    ax.set(title='Binned empirical cumulative probability',xlabel=col,ylabel='Cumulative probability',ylim=(-.03,1.07)); ax.grid(ls=':',alpha=.7); ax.legend(); render(fig)
    st.dataframe(pd.DataFrame({'Percentile':LABELS,'Raw-data value':raw_q,'Binned estimate':bq}),hide_index=True,use_container_width=True)
    st.caption('Binned percentile estimates interpolate within bins; raw-data percentiles are calculated from original observations and can differ. For more than 80 bins, bar labels are suppressed; the complete values are in the table.')
with cdf:
    st.subheader('Raw empirical CDF and all selected fitted CDFs')
    sorted_x=np.sort(x); grid=np.linspace(min(0,float(x.min())),float(x.max())+.05*np.ptp(x),800)
    fig,ax=plt.subplots(figsize=(12,6))
    ax.step(sorted_x,np.arange(1,len(x)+1)/len(x),where='post',lw=2,label='Empirical CDF (raw observations)')
    palette=plt.get_cmap('tab10')
    for i,f in enumerate(fits):
        ax.plot(grid,f['dist'].cdf(grid,*f['params']),color=palette(i%10),lw=2,
                label=f"Fitted {f['name']} CDF" + (' (simulation selection)' if f['name']==selected else ''))
    percent_lines(ax,raw_q)
    ax.set(title='Empirical CDF versus all selected fitted CDFs',xlabel=col,ylabel='P(X ≤ x)',ylim=(-.03,1.07))
    ax.minorticks_on(); ax.grid(which='major',ls=':',color='#888888'); ax.grid(which='minor',ls=':',color='#BBBBBB',alpha=.7)
    ax.legend(); render(fig)
    st.dataframe(pd.DataFrame({'Percentile':LABELS,'Raw-data value':raw_q}),hide_index=True,use_container_width=True)
with mc:
    minimum,maximum=float(x.min()),float(x.max())
    st.subheader(f'Monte Carlo simulation using fitted {selected}')
    st.caption(f'Draws are conditioned on the observed range [{minimum:.3f}, {maximum:.3f}]. '
               'Overlaid curves are fitted PDFs conditioned on that same range.')
    grid_mc=np.linspace(minimum,maximum,800)
    palette=plt.get_cmap('tab10')
    def simulation_plot(f, draw_seed, color, show_percentiles=False):
        values=simulate(f,int(runs),int(draw_seed),minimum,maximum)
        mq=np.percentile(values,[10,50,90])
        fig,ax=plt.subplots(figsize=(12,5))
        ax.hist(values,bins=min(60,max(10,int(np.sqrt(runs)))),range=(minimum,maximum),
                density=True,color=color,alpha=.42,edgecolor='black',label='Simulated density')
        ax.plot(grid_mc,conditional_pdf(f,grid_mc,minimum,maximum),color=color,lw=2.5,
                label=f"{f['name']} fitted PDF (conditioned)")
        if show_percentiles: percent_lines(ax,mq)
        ax.set(title=f"Monte Carlo: {f['name']} ({runs:,} runs)",xlabel=col,ylabel='Probability density',xlim=(minimum,maximum))
        ax.legend(); render(fig)
        return mq
    try:
        mq=simulation_plot(fit,int(seed),palette(names.index(selected)%10),show_percentiles=True)
        st.dataframe(pd.DataFrame({'Monte Carlo percentile':LABELS,'Simulated value':mq}),hide_index=True,use_container_width=True)
    except ValueError as e: st.error(str(e))
    st.subheader('Additional Monte Carlo plots for each fitted PDF')
    for i,f in enumerate(fits):
        try:
            st.markdown(f"**{f['name']}**")
            simulation_plot(f,int(seed)+i,palette(i%10))
        except ValueError as e: st.warning(str(e))
    st.caption('Monte Carlo percentiles come from simulated draws, not histogram bars. '
               'Conditioning on the observed range changes the fitted distribution and its percentiles.')
with gof:
    st.dataframe(pd.DataFrame([{'Distribution':f['name'],'KS statistic':f['ks'],'KS p-value*':f['pvalue'],'AIC':f['aic'],'BIC':f['bic']} for f in fits]).sort_values('KS statistic'),hide_index=True,use_container_width=True)
    st.caption('*KS p-values after fitting parameters on the same data are not calibrated fixed-parameter test p-values. Compare plots and physical suitability as well as statistics.')
