"""All report figures, traceable to saved raw results; vector PDF plus PNG."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9','#555555']


def save(fig,path):
    fig.savefig(str(path)+'.pdf',bbox_inches='tight')
    fig.savefig(str(path)+'.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def band(ax,frame,metric,label,color,style='-'):
    g=frame.groupby('cq')[metric]
    mean=g.mean();se=g.std()/np.sqrt(g.count())
    ax.plot(mean.index,mean,label=label,color=color,linestyle=style,lw=1.5)
    ax.fill_between(mean.index,mean-1.96*se,mean+1.96*se,color=color,alpha=.10)


def make_figures(output):
    output=Path(output);figdir=output/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    raw=pd.read_csv(output/'simulation_raw.csv')
    metrics=['r2','timing_mean','sharpe','norm2'];labels=['Out-of-sample R² (zero benchmark)','Mean timing return','Paper Sharpe (uncentered)','Squared coefficient norm']
    for strength in sorted(raw.strength.unique()):
        subset=raw[raw.strength==strength]
        for model in ('ridge','lasso'):
            fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
            groups=list(subset[subset.model==model].groupby('penalty'))
            for j,(penalty,frame) in enumerate(groups):
                for ax,metric in zip(axes.flat,metrics):
                    band(ax,frame,metric,('z=' if model=='ridge' else 'α=')+f'{penalty:g}',COLORS[j%len(COLORS)],['-','--','-.',':'][j%4])
            oracle=subset[subset.model=='ridge_oracle']
            for ax,metric,label in zip(axes.flat,metrics,labels):
                band(ax,oracle,metric,'DGP-known ridge reference','#111111','--')
                ax.axvline(1,color='#777777',linestyle=':',lw=.8)
                ax.set_xscale('log');ax.set_xlabel('Observed complexity cᵩ = P₁ / T_train');ax.set_ylabel(label)
                ax.grid(alpha=.18)
            if model=='ridge':
                axes[0,0].set_yscale('symlog',linthresh=.1)
                axes[1,1].set_yscale('log')
            axes[0,1].legend(fontsize=7,loc='upper left')
            save(fig,figdir/f'{model}_b{strength:g}')
    # Regularized R2 close-up makes gains visible despite the ridgeless peak.
    strengths=sorted(raw.strength.unique())
    fig,axes=plt.subplots(1,len(strengths),figsize=(5*len(strengths),3.4),squeeze=False,layout='constrained')
    for ax,strength in zip(axes.flat,strengths):
        sub=raw[raw.strength==strength]
        for j,z in enumerate([2,10,50]):
            band(ax,sub[(sub.model=='ridge')&(sub.penalty==z)],'r2',f'z={z}',COLORS[j])
        band(ax,sub[sub.model=='ridge_oracle'],'r2','DGP-known reference','#111111','--')
        ax.set_xlabel('Observed complexity cᵩ');ax.set_ylabel(f'OOS R², b*={strength:g}');ax.axhline(0,color='gray',lw=.6);ax.legend(fontsize=7);ax.grid(alpha=.2)
    save(fig,figdir/'ridge_regularized')
    rank=pd.read_csv(output/'validation_ranking.csv')
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
    for ax,pair in zip(axes,'ABC'):
        sub=rank[(rank.pair==pair)&(rank.stage=='final_all_train')].copy()
        sub['family']=sub.model.str.split('_').str[0]
        best=sub.sort_values('mse').groupby('family',sort=False).head(1)
        ax.barh(best.model,best.r2,color=COLORS[0]);ax.axvline(0,color='gray',lw=.7)
        ax.set_xlabel(f'Pair {pair}: selection-CV R²');ax.tick_params(axis='y',labelsize=6.5)
    save(fig,figdir/'prediction_validation')
    dyn=pd.read_csv(output/'dynamics_raw.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,6.3),layout='constrained')
    for j,width in enumerate(sorted(dyn.width.unique())):
        sub=dyn[dyn.width==width]
        for split,style in [('train',':'),('validation','--'),('stationary','-')]:
            g=sub[sub.split==split].groupby('time').mse.mean()
            axes[0,0].plot(g.index,g,color=COLORS[j],ls=style,label=f'm={width}, {split}')
        for split,style in [('stationary','-'),('shifted','--')]:
            g=sub[sub.split==split].groupby('time').sharpe.mean()
            axes[0,1].plot(g.index,g,color=COLORS[j],ls=style,label=f'm={width}, {split}')
        sub=sub[sub.split=='train']
        for ax,col in [(axes[1,0],'output_l1'),(axes[1,1],'alignment')]:
            g=sub.groupby('time')[col].mean();ax.plot(g.index,g,color=COLORS[j],label=f'm={width}')
    for ax,label in zip(axes.flat,['Mean squared error','Paper Sharpe','Second-layer L1 norm: mean |aᵢ|','Mean absolute latent alignment']):
        ax.set_xlabel('Scaled Euler time');ax.set_ylabel(label);ax.legend(fontsize=6.5);ax.grid(alpha=.2)
    save(fig,figdir/'neural_dynamics')
