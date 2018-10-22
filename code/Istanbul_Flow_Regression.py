#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 23:05:48 2018

@author: kai
"""
import pandas as pd
import scipy
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import networkx as nx
import math
import copy
import folium

import seaborn as sns
from statsmodels.formula.api import ols
sns.set(color_codes=True)
sns.set_style("whitegrid")
#%%


#helper to read raw credit card data , repeat means we consider more than one trip by a user
def readRawFlows(isRepeat):
    raw=pd.read_csv('istanbulData/raw/SU_ORNEKLEM_KK_HAR_BILGI.txt')
    raw=raw[(raw['UYEISYERI_ID_MASK']!=999999) & (raw.ONLINE_ISLEM ==0)]
    raw.rename(columns={'MUSTERI_ID_MASK': 'customer_id','UYEISYERI_ID_MASK': 'shop_id'},inplace=True)

    df2=pd.read_csv('istanbulData/raw/5-shopiddistid.csv')
   
    raw=pd.merge(raw,df2,on='shop_id',how='left')
    df=pd.read_csv('istanbulData/raw/6-custidhdistid.csv')
    raw=pd.merge(raw,df,on='customer_id',how='left')
    if not isRepeat:
        raw.drop_duplicates(subset=['shop_id','customer_id'],inplace=True)
    raw.rename(columns={'district_id':'sdistrict_id'},inplace=True)
    
    return raw

# helper to get customer information
def getCust():
    #62392 unique customers
    demog=pd.read_csv('istanbulData/raw/customers_60k_DEMOGRAPHICS.csv')
    demog['agecat']=(demog.age/10).astype(int)
    mask=demog.income==0
    avgincome=demog.income[demog.income!=0].mean()
    demog.loc[mask,'income']=avgincome
    demog['incomecat']=pd.qcut(demog.income,10,labels=False).tolist()
    hdf1=pd.read_csv('istanbulData/raw/6-custidhdistid.csv')
    demog=pd.merge(demog,hdf1,how='left',on='customer_id')
    wdf1=pd.read_csv('istanbulData/raw/7-custidwdistid.csv')
    demog=pd.merge(demog,wdf1,how='left',on='customer_id')
    demog.dropna(inplace=True)
    return demog


# helper to get flow matrix. hdistrict_id = origin. sdistrict_id = destination
def readInOutFlows(isRepeat):
    raw=readRawFlows(isRepeat)
    movement=raw.groupby(['hdistrict_id','district_id']).apply(len)
    movement=movement.reset_index()
    movement.rename(columns={0:'counts','district_id':'sdistrict_id'},inplace=True)
    return movement


# helper to get inflow demographic diversity,  and outflow destination diversity
def readInOutDiversity():
    demog=getCust()
    df=readRawFlows(repeat=True)
    df=pd.merge(df,demog,on='customer_id',how='left')
    df.dropna(subset=['hdistrict_id'],inplace=True)
    def diversity(chunk):
        return scipy.stats.entropy(chunk.counts.values)
    def total(chunk):
        return chunk.counts.sum()   

    #'vector' that describes the diversity of the people
#==============================================================================
#     df['composite_in']=df.agecat.astype(str) + df.incomecat.astype(str)+ df.hdistrict_id.astype(str) + df.education.astype(str) + df.gender.astype(str)
#==============================================================================
    df['composite_in']=df.agecat.astype(str) + df.incomecat.astype(str)+ df.wdistrict_id.astype(str)+ df.hdistrict_id.astype(str) + df.education.astype(str) + df.gender.astype(str)
#==============================================================================
#      df['composite_in']=df.agecat.astype(str) + df.incomecat.astype(str)+ df.hdistrict_id.astype(str) + df.education.astype(str) + df.gender.astype(str)
#==============================================================================
    groups=df.groupby(['sdistrict_id','composite_in']).apply(len)
    groups=groups.reset_index()
    groups.rename(columns={0:'counts','district_id':'sdistrict_id'},inplace=True)
    
    entros_in= groups.groupby(['sdistrict_id']).apply(diversity)
    entros_in=entros_in.reset_index()
    entros_in.rename(columns={0:'entropy_in'},inplace=True)
    entros_in['district_id']=entros_in.sdistrict_id
    
    groups=df.groupby(['sdistrict_id','hdistrict_id']).apply(len)
    groups=groups.reset_index()
    groups.rename(columns={0:'counts','district_id':'sdistrict_id'},inplace=True)
    
    
    entros_out= groups.groupby(['hdistrict_id']).apply(diversity)
    entros_out=entros_out.reset_index()
    entros_out.rename(columns={0:'entropy_out'},inplace=True)

    ddf2=entros_out.rename(columns={'hdistrict_id':'district_id'})
    ddf2=pd.merge(ddf2,entros_in,how='left',on='district_id')
    ddf2.drop(['sdistrict_id'],axis=1,inplace=True)
    ddf2.dropna(inplace=True) 
    ddf2.district_id=ddf2.district_id.astype(int)
    ddf2.to_csv('istanbulData/flow_diversity.csv')
    return ddf2


# diversity of avaialblity of shops - ie not weighted by consumption value 
def readAvailablityDiversity():

    df=readRawFlows(isRepeat=True)
    
    df2=pd.read_csv("istanbulData/raw/4-shopid_mccmerged.csv")
    
    df=pd.merge(df,df2,how="left",on="shop_id")
    
    df.drop_duplicates(subset = ['shop_id'],inplace=True)
    
    df=df[df.shop_id!=999999]
    
    def diversity(chunk):
        return scipy.stats.entropy(chunk.counts.values)
    def numUniqueDiversity(chunk):
        return len(chunk) 
    groups=df.groupby(['sdistrict_id','mcc_merged']).apply(len)
    groups=groups.reset_index()
    groups.rename(columns={0:'counts','district_id':'sdistrict_id'},inplace=True)

    entros_in= groups.groupby(['sdistrict_id']).apply(numUniqueDiversity)
    entros_in=entros_in.reset_index()
    entros_in.rename(columns={0:'mcc_merged'},inplace=True)
    entros_in['district_id']=entros_in.sdistrict_id
    
    groups=df.groupby(['sdistrict_id','mcc_detailed']).apply(len)
    groups=groups.reset_index()
    groups.rename(columns={0:'counts','district_id':'sdistrict_id'},inplace=True)
    
    
    entros_out= groups.groupby(['sdistrict_id']).apply(numUniqueDiversity)
    entros_out=entros_out.reset_index()
    entros_out.rename(columns={0:'mcc_detailed'},inplace=True)
    
    
    ddf2=entros_out.rename(columns={'sdistrict_id':'district_id'})
    ddf2=pd.merge(ddf2,entros_in,how='left',on='district_id')
    
    ddf2.drop(['sdistrict_id'],axis=1,inplace=True)
    
    
    ddf2.dropna(inplace=True)
    
    ddf2.district_id=ddf2.district_id.astype(int)
    ddf2.to_csv('istanbulData/shopavaildiversity.csv')
    return ddf2

#build network based on flow matrix. returns        
def buildSimpleNetwork(df):
    def mapp(dic,name):
        for key ,value in dic.iteritems():
            ddf[key][name]=value
                
    ddf={}
    
    G=nx.DiGraph()
    GI=nx.DiGraph()
    nodes=df.hdistrict_id.unique()
    for n in nodes:
        ddf[n]={}
    names=[]
    dics=[]
        
    G.add_nodes_from(nodes)
    for index,row in df.iterrows():
        G.add_edge(row.hdistrict_id,row.sdistrict_id,{'weight':row.counts,'distance':1.0/row.counts})
        GI.add_edge(row.sdistrict_id,row.hdistrict_id,{'weight':row.counts,'distance':1.0/row.counts})
    dics.append(nx.eigenvector_centrality(G,weight='weight'))
    names.append('eigen')
    
    dics.append(nx.eigenvector_centrality(GI,weight='weight'))
    names.append('righteigen')
    
    dics.append(nx.in_degree_centrality(G))
    names.append('indegree')
    
    dics.append(nx.out_degree_centrality(G))
    names.append('outdegree')
    
    dics.append(nx.closeness_centrality(G,distance='distance'))
    names.append('closeness')
    dics.append(nx.betweenness_centrality(G,weight='weight'))
    names.append('betweeness')
        
    def get(chunk):
        return chunk.counts.sum()
    dics.append(df.groupby('hdistrict_id').apply(get).to_dict())
    names.append('outgoing')
    dics.append(df.groupby('sdistrict_id').apply(get).to_dict())
    names.append('incoming')
    
    dic={}
    for n in nodes:
        neigh=G.neighbors(n)
        neigh.remove(n)
        N=len(neigh)
        s=0.0
        for i in range(len(neigh)):
            for j in range(i+1,len(neigh)):
                s+=G.has_edge(*(neigh[i],neigh[j]))
                s+=G.has_edge(*(neigh[j],neigh[i]))
        dic[n]=s/(N*(N-1))
    dics.append(dic)
    names.append('clustering')
    
    def check(row):
        return row.hdistrict_id==row.sdistrict_id
    df['check']=df.apply(check,1) 
    df2=df[df.check==False]
    dics.append(df2.groupby('hdistrict_id').apply(get).to_dict())
    names.append('outgoing_noself')
    dics.append(df2.groupby('sdistrict_id').apply(get).to_dict())
    names.append('incoming_noself')   
    for i in range(len(dics)):
        mapp(dics[i],names[i])
    ddf=pd.DataFrame.from_dict(ddf,orient='index') 
    ddf['district_id']=ddf.index
    return ddf
                
    

def regress(df):
    xvars=['incoming_noself','outgoing_noself']
#==============================================================================
#     xvars=['incoming','outgoing']
#==============================================================================
    yVar='delta1416p'
    yVar='y2016'
    df['logy2016']=np.log(df.y2016)
#==============================================================================
#     yVar='logy2016'
#==============================================================================
    xC='+'.join(xvars)
    model = ols("{} ~ {} ".format(yVar,xC),df).fit()
    print model.summary()
    

#functions for exploratory purposes SI

def getFlowMHeatMap():
    
    
    df_names=pd.read_csv('istanbulData/raw/Istanbul_district_area.txt')
    df_diverse=pd.read_csv('istanbulData/shopavaildiversity.csv')
    df_diverse=pd.merge(df_diverse,df_names,on='district_id',how='left')
    df_diverse= df_diverse.sort_values('district_id')
    
    districts=np.sort(df_diverse.district_id.values.tolist())
    
    df=readRawFlows(True)
    df=df.dropna(subset=['hdistrict_id','sdistrict_id'])
    df2=df.groupby(['hdistrict_id','sdistrict_id']).apply(len)
    df2=df2.reset_index()

    mapper={}
    demapper={}
    matrix=np.zeros(shape=(len(districts),len(districts)))
    for i in range(len(districts)):
        demapper[i]=districts[i]
        mapper[districts[i]]=i
    
    def addToMat(row):
        
        try:
            fromm=mapper[row.hdistrict_id]
            to=mapper[row.sdistrict_id]
            if to!=fromm:
                matrix[fromm,to]=row[0]
        except:
            pass
    df2.apply(addToMat,1)
    
    plt.figure()
#==============================================================================
#     sns.heatmap(matrix)
#     plt.yticks(range(len(df_diverse)),df_diverse.district_name.values.tolist())
#     plt.xticks(range(len(df_diverse)),df_diverse.district_name.values.tolist(),rotation='vertical',verticalalignment='top')
#     
#==============================================================================
    plt.savefig('alalala',dpi=300)
    sns.heatmap(df_diverse.mcc_detailed.values.reshape(39,1),cmap='PuBu',cbar_kws = dict(use_gridspec=False,location="top"))
def getHouseAndPopDseMap():
    df=pd.read_csv("thesisData/beijing_cc.csv")
    df2=pd.read_csv("yelp/istanbul_cc_n.csv")
    df3=pd.read_csv("thesisData/usa_cc.csv")
    x="popdse"
    y="HPI"
    df=df[df.HPI!=0]
    mapCreate(df2,"istan_HPI",y)
    mapCreate(df2,"istan_popdse",x)
#==============================================================================
#     plot(df3,x,y)
#==============================================================================
    
    
def plot(df,xVar,yVar,control=False):
    feature=copy.deepcopy(df)
    
    xVarsC=[u'popdse','weightedEigenCentralityDist','HPI']
    plt.figure()
    xlab= 'Population Density'
    ylab= "Diversity of Flow (in Thousands)"
    plt.subplot(111)
    plt.scatter(feature[xVar].values, feature[yVar].values,color='#1B5F68')
    x=feature[xVar].values
    y=feature[yVar].values
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.xlabel(xlab,size=15)
    plt.ylabel(ylab,size=15)
    ans=feature[[xVar, yVar]].corr()[yVar][xVar]
#==============================================================================
#     plt.text(6,2.1,'Correlation : {}'.format(ans),size=15)
#==============================================================================
    plt.ylabel(ylab,size=15)
    plt.savefig('istanplot1.png')
#==============================================================================
#     plt.title('No control')
#==============================================================================
    plt.savefig('istanplot2.png')
    

def plot2(xVar,yVar):
    feature=pd.read_csv('istanbulData/raw/allflows.csv')
    feature['totalflow']=feature.Inflow+feature.outflow
    df=pd.read_csv('istanbulData/raw/gdp.csv')
    feature=pd.merge(feature,df,on='district_id',how="left")
    plt.figure()
#==============================================================================
#     xlab='Number of unique shop categories'
#==============================================================================
    xlab= 'Diversity of People entering in district'
    ylab= "Diversity of Flow (in Thousands)"
    ylab= "Economic Productivity"
    xlab='Volume of Total Flow'
    plt.subplot(111)
    plt.scatter(feature[xVar].values, feature[yVar].values,color='#FC63A8')
    x=feature[xVar].values
    y=feature[yVar].values
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.ylim(bottom=0)
    plt.xlabel(xlab,size=15)
    plt.ylabel(ylab,size=15)
    ans=feature[[xVar, yVar]].corr()[yVar][xVar]
    plt.text(1000,1600000000,'Correlation : {}'.format(ans),size=15)
    plt.ylabel(ylab,size=15)
    plt.savefig('istanplot1.png')


def mapCreate(df,name,v):
    df=df.append(df[df.district_id==115])
    df.iloc[-1,df.columns.get_loc('district_id')]=116.0
    state_geo = 'istanbulData/district_level_shape/district.geojson'
#==============================================================================
#     state_geo = 'td/bj_shapefile/bj_shapefile.geojson'
#==============================================================================
    # Initialize the map:
    m = folium.Map(location=[41.0082, 28.9784], zoom_start=10,tiles='cartodbpositron')
     
    # Add the color for the chloropleth:
    style_function = lambda x: {'fillOpacity': 1 if       
                            x['properties']['districtid']==0 else
                             0, 'fillColor' : "#000000",'stroke': False }

    mapbg=folium.GeoJson(state_geo,style_function)
    
    m.choropleth(
     geo_data=state_geo,
     name='choropleth',
     data=df,
     columns=['district_id',v],
     key_on='feature.properties.districtid',
     fill_color='PuBu',
     fill_opacity=0.7,
     line_opacity=0.2,
     legend_name='Value'
    )
    mapbg.add_to(m)
    # Save to html
    m.save('sted_{}.html'.format(name))
    
if __name__ == "__main__":
    getHouseAndPopDseMap()
    pass

