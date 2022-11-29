# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:02:26 2021

@author: KjSno
"""
import pandas as pd
import numpy as np
import datetime
# from model_update import new_weather
import math

from sklearn.linear_model import LinearRegression

import os


yearsAvg = 20
countDate=datetime.date.today()-datetime.timedelta(days=1)
countDate=datetime.date(2022,11,24)
countDate=datetime.datetime(countDate.year,countDate.month,countDate.day)
allModels=pd.read_excel(r'J:\Trading\Quant\_Yield_Models\Models Running.xlsx')

# yieldHist=yields2
# weightedWeather=weather
# Dump=weatherDump
# modelInfo=model
# today=datetime.datetime.today()

"Years is based on Soybeans"
def model_update(yearsAvg,yieldHist,weightedWeather,Dump,modelInfo,today=datetime.datetime.today(),StretchBool=False):
        years=np.reshape(yieldHist['year'].to_numpy(),(-1,1))
        yields=np.reshape(yieldHist['Yield'].to_numpy(),(-1,1))
        
        firstDate=weightedWeather.index[0].to_pydatetime()
        
        forecastYear=modelInfo[1][2]
        
        firstDateNum=firstDate.toordinal()
        trendBreak=modelInfo[1][1]
        if trendBreak<=0:
            trendRegX=years
            newTrendRegX=np.reshape(forecastYear,(-1,1))
        else:
            trendRegX=np.zeros((len(years),3))
            trendRegX[:,0:1]=years
            trendRegX[:,1]=([years>=trendBreak]*years)[0,:,0]
            trendRegX[:,2]=np.array([years>=trendBreak])[0,:][:,0]
            newTrendRegX=np.reshape([forecastYear,forecastYear,1],(1,3))
        trendReg=LinearRegression().fit(trendRegX,yields)
        trendYields=trendReg.predict(trendRegX)
        yieldVsTrend=yields/trendYields-1
            
        newTrendYield=trendReg.predict(newTrendRegX)
        
        modelInfo = modelInfo.replace("Soil Moist","soil_moist_15")
        modelInfo = modelInfo.replace("Temp","temp")
        modelInfo = modelInfo.replace("Soil Moist Squared","soil_moist_15_sq")
        
        modelInfo=pd.DataFrame(modelInfo.to_numpy()[7:,1:],columns=modelInfo.to_numpy()[7,1:])
        
        
        
        Dump=Dump.reset_index().drop(columns=['area'])
        
        Dump['Together (real+eu)']=Dump['Together (real+eu)'].astype(float)
        EUDump=pd.pivot_table(Dump,values='Together (real+eu)',index='weatherDate',columns='metric')
        realDump=pd.pivot_table(Dump,values='Realized',index='weatherDate',columns='metric')
        GFSDump=Dump
        GFSDump['GFS']=GFSDump['Realized'].fillna(GFSDump['GFS'])
        GFSDump=pd.pivot_table(GFSDump,values='GFS',index='weatherDate',columns='metric')    
        
        weightedWeather.index=pd.to_datetime(weightedWeather.index)
            
        weightedWeather=weightedWeather[weightedWeather.index<today-datetime.timedelta(days=28)]
        EUDump=EUDump[EUDump.index>=today-datetime.timedelta(days=28)]
        realDump=realDump[realDump.index>=today-datetime.timedelta(days=28)]
        GFSDump=GFSDump[GFSDump.index>=today-datetime.timedelta(days=28)]
    
    
    
    
        addMatrix=np.zeros((300,len(weightedWeather.columns)))
        dateVec=np.empty(len(weightedWeather)+300,dtype=object)
        dateVec[:len(weightedWeather)]=weightedWeather.index.to_numpy()
        startDate=weightedWeather.index[-1]+datetime.timedelta(days=1)
        weightedWeather2=weightedWeather.to_numpy()
    
    
    
        nextYear=False
        for i in range (0,300):
            newDate=startDate+datetime.timedelta(days=i)
            if(newDate.month==firstDate.month and newDate.day==firstDate.day):
                nextYear=True
            dateVec[len(weightedWeather)+i]=newDate
            if(newDate.month==2 and newDate.day==29):
                newDate=newDate-datetime.timedelta(days=1)
            if(nextYear):
                for j in range(2,yearsAvg+2):
                    avgDate=datetime.date(newDate.year-j,newDate.month,newDate.day).toordinal()-firstDateNum
                    addMatrix[i,:]=addMatrix[i,:]+weightedWeather2[avgDate,:]/yearsAvg
            else:      
                for j in range(1,yearsAvg+1):
                    avgDate=datetime.date(newDate.year-j,newDate.month,newDate.day).toordinal()-firstDateNum
                    addMatrix[i,:]=addMatrix[i,:]+weightedWeather2[avgDate,:]/yearsAvg
                
        fullWeather=np.concatenate((weightedWeather2,addMatrix))
    
    
    
        def addcolls(df):
            df=df.rename(columns={"Precipitation":"Precip","Soil moisture at -15cm":"Soil_moist_15",\
                                  "Maximum temperature":"Max_temp","Mean temperature":"Temp","Snow depth":"snow_depth","Minimum temperature":"min_temp"})
            try:
                df=df[['Precip', 'Temp', 'Soil_moist_15', 'Max_temp','snow_depth']]
            except:
                df['snow_depth']=0
                df=df[['Precip', 'Temp', 'Soil_moist_15', 'Max_temp','snow_depth']]
            return df
       
        def expandweather(df,weather):
            startPoint=df.index[0].toordinal()-firstDateNum
            endPoint=df.index[-1].toordinal()-firstDateNum
            df=df.to_numpy()
            for i in range(len(df)):
                for j in range(len(df[0,:])):
                    if np.isnan(df[i,j]):
                        df[i,j]=fullWeather[startPoint+i,j]
            newWeather=np.zeros((len(weather),len(weather[0,:])))
            for i in range(len(newWeather)):
                for j in range(len(newWeather[0,:])):
                    if j<4:
                        if i >=startPoint and i<=endPoint:
                            newWeather[i,j]=df[i-startPoint,j]
                        else:
                            newWeather[i,j]=weather[i,j]
                    else:
                        if i >=startPoint and i<=endPoint:
                            newWeather[i,j]=df[i-startPoint,j-4]*df[i-startPoint,j-4]
                        else:
                            newWeather[i,j]=weather[i,j-4]*weather[i,j-4]
            return newWeather
        print(firstDate)
    
        EUDump=addcolls(EUDump)
        EUDump=expandweather(EUDump,fullWeather)
                
        realDump=addcolls(realDump)
        realDump=expandweather(realDump,fullWeather)
        
        GFSDump=addcolls(GFSDump)
        GFSDump=expandweather(GFSDump,fullWeather)
        
        if StretchBool:
            stretch=max(12,int((datetime.datetime(2023,1,15)-today).days))
        else:
            
            stretch=12
     
        for j in range(1,stretch):
            val=0
            now=today-datetime.timedelta(days=0)
            for i in range(1,yearsAvg+1):
                val=val+EUDump[datetime.date(now.year-i,now.month,now.day).toordinal()+9-firstDateNum,2]/EUDump[datetime.date(now.year-i,now.month,now.day).toordinal()+9-firstDateNum+j,2]/yearsAvg
            EUDump[datetime.date(now.year,now.month,now.day).toordinal()+j+9-firstDateNum,2]=EUDump[datetime.date(now.year,now.month,now.day).toordinal()+j+9-firstDateNum,2]*(j)/(stretch)+EUDump[datetime.date(now.year,now.month,now.day).toordinal()+9-firstDateNum,2]/val*(stretch-j)/stretch
        
        for j in range(1,stretch):
            val=0
            now=today-datetime.timedelta(days=0)
            for i in range(1,yearsAvg+1):
                val=val+GFSDump[datetime.date(now.year-i,now.month,now.day).toordinal()-firstDateNum,2]/GFSDump[datetime.date(now.year-i,now.month,now.day).toordinal()-firstDateNum+j,2]/yearsAvg
            GFSDump[datetime.date(now.year,now.month,now.day).toordinal()+j-firstDateNum,2]=GFSDump[datetime.date(now.year,now.month,now.day).toordinal()-firstDateNum+j,2]*(j)/(stretch)+GFSDump[datetime.date(now.year,now.month,now.day).toordinal()-firstDateNum,2]/val*(stretch-j)/stretch
            realDump[datetime.date(now.year,now.month,now.day).toordinal()+j-firstDateNum,2]=realDump[datetime.date(now.year,now.month,now.day).toordinal()-firstDateNum+j,2]*(j)/(stretch)+realDump[datetime.date(now.year,now.month,now.day).toordinal()-firstDateNum,2]/val*(stretch-j)/stretch
    
    
    
    
        
        
        modelInfo=modelInfo.dropna(axis=1,how='all').to_numpy()
        
        if(trendBreak==0):
            realVars=np.ones((len(years),len(modelInfo[0,:])+1))*-2
            realVars[:,-1:]=trendRegX
            
            realGFSVars=np.ones((len(years),len(modelInfo[0,:])+1))*-2
            realGFSVars[:,-1:]=trendRegX
            
            realVars=np.ones((len(years),len(modelInfo[0,:])+1))*-2
            realVars[:,-1:]=trendRegX
            
            GFSVars=np.ones((len(years),len(modelInfo[0,:])+1))*-2
            GFSVars[:,-1:]=trendRegX
            
            EUVars=np.ones((len(years),len(modelInfo[0,:])+1))*-2
            EUVars[:,-1:]=trendRegX
            
            
            
            
            realVarNew=np.zeros((1,len(modelInfo[0,:])+1))
            realVarNew[:,-1:]=forecastYear
            
            EUVarNew=np.zeros((1,len(modelInfo[0,:])+1))
            EUVarNew[:,-1:]=forecastYear
            
            GFSVarNew=np.zeros((1,len(modelInfo[0,:])+1))
            GFSVarNew[:,-1:]=forecastYear
            
            realGFSVarNew=np.zeros((1,len(modelInfo[0,:])+1))
            realGFSVarNew[:,-1:]=forecastYear
            
        else:
            realVars=np.ones((len(years),len(modelInfo[0,:])+3))*-2
            realVars[:,-3:]=trendRegX
            
            realGFSVars=np.ones((len(years),len(modelInfo[0,:])+3))*-2
            realGFSVars[:,-3:]=trendRegX
            
            EUVars=np.ones((len(years),len(modelInfo[0,:])+3))*-2
            EUVars[:,-3:]=trendRegX
            
            GFSVars=np.ones((len(years),len(modelInfo[0,:])+3))*-2
            GFSVars[:,-3:]=trendRegX
        
            realVarNew=np.zeros((1,len(modelInfo[0,:])+3))
            realVarNew[:,-3]=forecastYear
            realVarNew[:,-2]=forecastYear
            realVarNew[:,-1]=1
            
        
            EUVarNew=np.zeros((1,len(modelInfo[0,:])+3))
            EUVarNew[:,-3]=forecastYear
            EUVarNew[:,-2]=forecastYear
            EUVarNew[:,-1]=1
        
            GFSVarNew=np.zeros((1,len(modelInfo[0,:])+3))
            GFSVarNew[:,-3]=forecastYear
            GFSVarNew[:,-2]=forecastYear
            GFSVarNew[:,-1]=1
        
            realGFSVarNew=np.zeros((1,len(modelInfo[0,:])+3))
            realGFSVarNew[:,-3]=forecastYear
            realGFSVarNew[:,-2]=forecastYear
            realGFSVarNew[:,-1]=1
        
        weatherColumns=['precip','temp','soil_moist_15','max_temp','precip_sq','temp_sq','soil_moist_15_sq','max_temp_sq','temp_below']
        # weatherColumns=['precip','temp','soil_moist_15','max_temp','min_temp','precip_sq','temp_sq','soil_moist_15_sq','max_temp_sq','temp_below']
        for i in range(len(modelInfo[0,:])):
            if modelInfo[0,i]=='temp_below':
                for j in years[:,0]:
                    j2=int(j)
                    col=weatherColumns.index('temp')
                    EUVars[int(j-years[0,0]),i]=np.average((EUDump[datetime.datetime(j2+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(j2+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]<modelInfo[4,1])*1)
                    realVars[int(j-years[0,0]),i]=np.average((realDump[datetime.datetime(j2+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(j2+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]<modelInfo[4,1])*1)

                    GFSVars[int(j-years[0,0]),i]=np.average((GFSDump[datetime.datetime(j2+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(j2+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]<modelInfo[4,1])*1)
                    realGFSVars[int(j-years[0,0]),i]=np.average((realDump[datetime.datetime(j2+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(j2+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]<modelInfo[4,1])*1)
                
                EUVarNew[0,i]=np.average((EUDump[datetime.datetime(forecastYear+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(forecastYear+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]<modelInfo[4,1])*1)
                realVarNew[0,i]=np.average((realDump[datetime.datetime(forecastYear+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(forecastYear+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]<modelInfo[4,1])*1)
                GFSVarNew[0,i]=np.average((GFSDump[datetime.datetime(forecastYear+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(forecastYear+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]<modelInfo[4,1])*1)
                realGFSVarNew[0,i]=np.average((realDump[datetime.datetime(forecastYear+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(forecastYear+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]<modelInfo[4,1])*1)
            elif modelInfo[0,i] == 'temp_above':
                for j in years[:,0]:
                    j2=int(j)
                    col=weatherColumns.index('temp')
                    EUVars[int(j-years[0,0]),i]=np.average((EUDump[datetime.datetime(j2+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(j2+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]>modelInfo[3,1])*1)
                    realVars[int(j-years[0,0]),i]=np.average((realDump[datetime.datetime(j2+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(j2+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]>modelInfo[3,1])*1)

                    GFSVars[int(j-years[0,0]),i]=np.average((GFSDump[datetime.datetime(j2+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(j2+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]>modelInfo[3,1])*1)
                    realGFSVars[int(j-years[0,0]),i]=np.average((realDump[datetime.datetime(j2+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(j2+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]>modelInfo[3,1])*1)
                
                EUVarNew[0,i]=np.average((EUDump[datetime.datetime(forecastYear+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(forecastYear+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]>modelInfo[3,1])*1)
                realVarNew[0,i]=np.average((realDump[datetime.datetime(forecastYear+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(forecastYear+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]>modelInfo[3,1])*1)

                GFSVarNew[0,i]=np.average((GFSDump[datetime.datetime(forecastYear+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(forecastYear+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]>modelInfo[3,1])*1)
                realGFSVarNew[0,i]=np.average((realDump[datetime.datetime(forecastYear+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(forecastYear+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col]>modelInfo[3,1])*1)


            elif modelInfo[0,i] == 'Dummy':
                for j in years[:,0]:
                    j2=int(j)
                    col=weatherColumns.index('temp')
                    if(years[j-years[0,0],0]==modelInfo[1,i]):
                        
                        EUVars[int(j-years[0,0]),i]=1
                        realVars[int(j-years[0,0]),i]=1
                        
                        GFSVars[int(j-years[0,0]),i]=1
                        realGFSVars[int(j-years[0,0]),i]=1
                    else:
                        EUVars[int(j-years[0,0]),i]=0
                        realVars[int(j-years[0,0]),i]=0
                        
                        GFSVars[int(j-years[0,0]),i]=0
                        realGFSVars[int(j-years[0,0]),i]=0
                EUVarNew[0,i]=0
                realVarNew[0,i]=0
                GFSVarNew[0,i]=0
                realGFSVarNew[0,i]=0
            else:
                col=weatherColumns.index(modelInfo[0,i])
                for j in years[:,0]:
                    j2=int(j)
                    EUVars[int(j-years[0,0]),i]=np.average(EUDump[datetime.datetime(j2+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(j2+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col])
                    realVars[int(j-years[0,0]),i]=np.average(realDump[datetime.datetime(j2+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(j2+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col])
                    if col in [2,6]:
                        GFSVars[int(j-years[0,0]),i]=np.average(GFSDump[datetime.datetime(j2+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(j2+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col-2])
                        realGFSVars[int(j-years[0,0]),i]=np.average(realDump[datetime.datetime(j2+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(j2+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col-2])
                    else:
                        GFSVars[int(j-years[0,0]),i]=np.average(GFSDump[datetime.datetime(j2+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(j2+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col])
                        realGFSVars[int(j-years[0,0]),i]=np.average(realDump[datetime.datetime(j2+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(j2+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col])
                
                EUVarNew[0,i]=np.average(EUDump[datetime.datetime(forecastYear+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(forecastYear+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col])
                realVarNew[0,i]=np.average(realDump[datetime.datetime(forecastYear+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(forecastYear+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col])
                if col in [2,6]:
                    GFSVarNew[0,i]=np.average(GFSDump[datetime.datetime(forecastYear+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(forecastYear+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col-2])
                    realGFSVarNew[0,i]=np.average(realDump[datetime.datetime(forecastYear+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(forecastYear+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col-2])
                else:
                    GFSVarNew[0,i]=np.average(GFSDump[datetime.datetime(forecastYear+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(forecastYear+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col])
                    realGFSVarNew[0,i]=np.average(realDump[datetime.datetime(forecastYear+int(math.floor((modelInfo[1,i]-1)/12)),int(math.fmod(modelInfo[1,i]-0.1,12)+0.1),1).toordinal()-1+modelInfo[2,i]-firstDateNum:datetime.datetime(forecastYear+int(math.floor((modelInfo[3,i]-1)/12)),int(math.fmod(modelInfo[3,i]-0.1,12)+0.1),1).toordinal()+modelInfo[4,i]-firstDateNum,col])

        "op dit moment zou het met negatieve temps nog fout kunnen gaan hieronder"
        
        print(realGFSVarNew)
        realGFSVars=realGFSVars[:,np.sum(realGFSVars,axis=0)>-1*len(realGFSVars)]
        realGFSReg=LinearRegression().fit(realGFSVars, yieldVsTrend)
        realGFSEst=realGFSReg.predict(realGFSVarNew)[0]
        
        GFSVars=GFSVars[:,np.sum(GFSVars,axis=0)>-1*len(GFSVars)]
        GFSReg=LinearRegression().fit(GFSVars, yieldVsTrend)
        GFSEst=GFSReg.predict(GFSVarNew)[0]
        print(GFSVarNew)
        
        EUVarNew=EUVarNew[:,np.sum(EUVars,axis=0)>-1*len(EUVars)]
        EUVars=EUVars[:,np.sum(EUVars,axis=0)>-1*len(EUVars)]
        EUReg=LinearRegression().fit(EUVars, yieldVsTrend)
        EUEst=EUReg.predict(EUVarNew)[0]
        print(EUVarNew)
    
        yieldVsTrend_cond=yieldVsTrend[np.sum(np.isnan(realVars),axis=1)==0]
        realVars=realVars[np.sum(np.isnan(realVars),axis=1)==0]
        RealReg=LinearRegression().fit(realVars, yieldVsTrend_cond)
        RealEst=RealReg.predict(realVarNew)[0]
        print(realVarNew)
        
        print(RealEst)
        print(GFSEst)
        print(realGFSEst)
        print(EUEst)
        # print(newTrendYield)
        
        
        EUDump=pd.DataFrame(EUDump,columns=weightedWeather.columns,index=pd.date_range(start=firstDate, periods=len(EUDump)))
           
        return EUDump,((RealEst+1)*newTrendYield),((GFSEst-realGFSEst+RealEst+1)*newTrendYield),((EUEst+1)*newTrendYield)


def createChart(filename, Yields):
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    Yields.to_excel(writer, sheet_name='Yields')
    workbook = writer.book
    worksheet = writer.sheets['Yields']
    chart = workbook.add_chart({'type': 'line'})
    colors = ['green', 'blue', 'red', 'cyan']
    for i in range(1, 4):
        chart.add_series({
            'name': ['Yields', 0, i + 1],
            'categories': ['Yields', 1, 0, len(Yields), 0],
            'values': ['Yields', 1, i + 1, len(Yields), i + 1],
            'line': {'width': 2, 'color': colors[i-1]},
        })
    chart.set_x_axis(
        {'name': 'Date', 'major_gridlines': {'visible': True, 'line': {'color': 'gray', 'transparency': 80}}})
    # print(np.min(Yields[['Realized','EU','GFS']].to_numpy()))
    chart.set_y_axis({'name': 'Value', 'min': math.floor((min(np.min(Yields[['Realized','EU','GFS']])) - 0.01) / 0.5) * 0.5,
                      'major_gridlines': {'visible': True, 'line': {'color': 'gray', 'transparency': 80}}})
    # chart.set_y_axis({'label_position': 'low'})
    chart.set_size({'width': 960, 'height': 600})
    chart.set_legend({'position': 'bottom'})
    chart.set_title({'name': 'Yields', }, )
    # worksheet.insert_chart('W2', chart, {'x_offset': 25, 'y_offset': 10})

    # Insert the chart into the worksheet.
    worksheet.insert_chart('F2', chart)
    writer.save()
    writer.close()
    
monthVec=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']



while(countDate<datetime.datetime.now()+datetime.timedelta(days=0)):
    print(countDate)
    month=monthVec[countDate.month-1]
    for i in allModels.index:
        if(allModels[month][i]==1):
            print(i)
            weather=pd.read_excel(allModels['weatherpath'][i],index_col=0)
            yields=pd.read_excel(allModels['historicyield'][i])
            yields2=yields
            model=pd.read_excel(allModels['modelpath'][i],header=None)
            weatherDump=pd.read_excel(allModels['dumppath'][i]+countDate.strftime("%Y-%m-%d")+'.xlsx')
            WeatherDump=weatherDump[weatherDump['area']==allModels['dumpregionname'][i]]
            outputFile=pd.read_excel(allModels['filepath'][i],index_col=0)
            outputFilename=allModels['filepath'][i]
            if(i==7):
                    weer,RealEst,GFSEst,EUEst=model_update(yearsAvg,yields,weather,weatherDump,model,today=countDate,StretchBool=True)
            else:
                weer,RealEst,GFSEst,EUEst=model_update(yearsAvg,yields,weather,weatherDump,model,today=countDate)
            weer.to_excel(allModels['weatherpath'][i])
            newrow=pd.DataFrame(np.reshape([model[1][5],RealEst[0][0],EUEst[0][0],GFSEst[0][0]],(1,4)),columns=outputFile.columns,index=[countDate.strftime("%m/%d/%Y")])
            # print(newrow)
            outputFile=outputFile.append(newrow)
            outputFile['EU']=pd.to_numeric(outputFile['EU'])
            outputFile['GFS']=pd.to_numeric(outputFile['GFS'])
            outputFile['Realized']=pd.to_numeric(outputFile['Realized'])
            createChart(allModels['filepath'][i],outputFile)
            
            print("real"+str(RealEst[0]))
            print("GFS"+str(GFSEst[0]))
            print("EU"+str(EUEst[0]))
            


    countDate+=datetime.timedelta(days=1)





