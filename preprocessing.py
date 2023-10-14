import numpy as np

columnsToKeep = [ '_RFHYPE5', 'TOLDHI2', '_CHOLCHK', 
                  '_BMI5', 'SMOKE100', 'CVDSTRK3', 
                  'DIABETE3', '_TOTINDA', '_FRTLT1',
                  '_VEGLT1', '_RFDRHV5', 'HLTHPLN1', 
                  'MEDCOST', 'GENHLTH', 'MENTHLTH', 
                  'PHYSHLTH', 'DIFFWALK', 'SEX',
                  '_AGEG5YR', 'EDUCA', 'INCOME2' ]

columns = ["_STATE","FMONTH","IDATE","IMONTH","IDAY","IYEAR","DISPCODE","SEQNO","_PSU","CTELENUM","PVTRESD1","COLGHOUS","STATERES","CELLFON3","LADULT","NUMADULT","NUMMEN","NUMWOMEN","CTELNUM1","CELLFON2","CADULT","PVTRESD2","CCLGHOUS","CSTATE","LANDLINE","HHADULT","GENHLTH","PHYSHLTH","MENTHLTH","POORHLTH","HLTHPLN1","PERSDOC2","MEDCOST","CHECKUP1","BPHIGH4","BPMEDS","BLOODCHO","CHOLCHK","TOLDHI2","CVDSTRK3","ASTHMA3","ASTHNOW","CHCSCNCR","CHCOCNCR","CHCCOPD1","HAVARTH3","ADDEPEV2","CHCKIDNY","DIABETE3","DIABAGE2","SEX","MARITAL","EDUCA","RENTHOM1","NUMHHOL2","NUMPHON2","CPDEMO1","VETERAN3","EMPLOY1","CHILDREN","INCOME2","INTERNET","WEIGHT2","HEIGHT3","PREGNANT","QLACTLM2","USEEQUIP","BLIND","DECIDE","DIFFWALK","DIFFDRES","DIFFALON","SMOKE100","SMOKDAY2","STOPSMK2","LASTSMK2","USENOW3","ALCDAY5","AVEDRNK2","DRNK3GE5","MAXDRNKS","FRUITJU1","FRUIT1","FVBEANS","FVGREEN","FVORANG","VEGETAB1","EXERANY2","EXRACT11","EXEROFT1","EXERHMM1","EXRACT21","EXEROFT2","EXERHMM2","STRENGTH","LMTJOIN3","ARTHDIS2","ARTHSOCL","JOINPAIN","SEATBELT","FLUSHOT6","FLSHTMY2","IMFVPLAC","PNEUVAC3","HIVTST6","HIVTSTD3","WHRTST10","PDIABTST","PREDIAB1","INSULIN","BLDSUGAR","FEETCHK2","DOCTDIAB","CHKHEMO3","FEETCHK","EYEEXAM","DIABEYE","DIABEDU","CAREGIV1","CRGVREL1","CRGVLNG1","CRGVHRS1","CRGVPRB1","CRGVPERS","CRGVHOUS","CRGVMST2","CRGVEXPT","VIDFCLT2","VIREDIF3","VIPRFVS2","VINOCRE2","VIEYEXM2","VIINSUR2","VICTRCT4","VIGLUMA2","VIMACDG2","CIMEMLOS","CDHOUSE","CDASSIST","CDHELP","CDSOCIAL","CDDISCUS","WTCHSALT","LONGWTCH","DRADVISE","ASTHMAGE","ASATTACK","ASERVIST","ASDRVIST","ASRCHKUP","ASACTLIM","ASYMPTOM","ASNOSLEP","ASTHMED3","ASINHALR","HAREHAB1","STREHAB1","CVDASPRN","ASPUNSAF","RLIVPAIN","RDUCHART","RDUCSTRK","ARTTODAY","ARTHWGT","ARTHEXER","ARTHEDU","TETANUS","HPVADVC2","HPVADSHT","SHINGLE2","HADMAM","HOWLONG","HADPAP2","LASTPAP2","HPVTEST","HPLSTTST","HADHYST2","PROFEXAM","LENGEXAM","BLDSTOOL","LSTBLDS3","HADSIGM3","HADSGCO1","LASTSIG3","PCPSAAD2","PCPSADI1","PCPSARE1","PSATEST1","PSATIME","PCPSARS1","PCPSADE1","PCDMDECN","SCNTMNY1","SCNTMEL1","SCNTPAID","SCNTWRK1","SCNTLPAD","SCNTLWK1","SXORIENT","TRNSGNDR","RCSGENDR","RCSRLTN2","CASTHDX2","CASTHNO2","EMTSUPRT","LSATISFY","ADPLEASR","ADDOWN","ADSLEEP","ADENERGY","ADEAT1","ADFAIL","ADTHINK","ADMOVE","MISTMNT","ADANXEV","QSTVER","QSTLANG","MSCODE","_STSTR","_STRWT","_RAWRAKE","_WT2RAKE","_CHISPNC","_CRACE1","_CPRACE","_CLLCPWT","_DUALUSE","_DUALCOR","_LLCPWT","_RFHLTH","_HCVU651","_RFHYPE5","_CHOLCHK","_RFCHOL","_LTASTH1","_CASTHM1","_ASTHMS1","_DRDXAR1","_PRACE1","_MRACE1","_HISPANC","_RACE","_RACEG21","_RACEGR3","_RACE_G1","_AGEG5YR","_AGE65YR","_AGE80","_AGE_G","HTIN4","HTM4","WTKG3","_BMI5","_BMI5CAT","_RFBMI5","_CHLDCNT","_EDUCAG","_INCOMG","_SMOKER3","_RFSMOK3","DRNKANY5","DROCDY3_","_RFBING5","_DRNKWEK","_RFDRHV5","FTJUDA1_","FRUTDA1_","BEANDAY_","GRENDAY_","ORNGDAY_","VEGEDA1_","_MISFRTN","_MISVEGN","_FRTRESP","_VEGRESP","_FRUTSUM","_VEGESUM","_FRTLT1","_VEGLT1","_FRT16","_VEG23","_FRUITEX","_VEGETEX","_TOTINDA","METVL11_","METVL21_","MAXVO2_","FC60_","ACTIN11_","ACTIN21_","PADUR1_","PADUR2_","PAFREQ1_","PAFREQ2_","_MINAC11","_MINAC21","STRFREQ_","PAMISS1_","PAMIN11_","PAMIN21_","PA1MIN_","PAVIG11_","PAVIG21_","PA1VIGM_","_PACAT1","_PAINDX1","_PA150R2","_PA300R2","_PA30021","_PASTRNG","_PAREC1","_PASTAE1","_LMTACT1","_LMTWRK1","_LMTSCL1","_RFSEAT2","_RFSEAT3","_FLSHOT6","_PNEUMO2","_AIDTST3"]

y = np.genfromtxt("dataset/y_train.csv", delimiter=",", skip_header=1, usecols=0)
x = np.genfromtxt("dataset/x_train.csv", delimiter=",", skip_header=1)

#Keep only the meaningful columns

indexColumnsToKeep = []
mapCols = dict()
newIndex = 0
for i,label in enumerate(columns):
  if label in columnsToKeep:
    for j,col in enumerate(columnsToKeep):    
      if label == col:
        indexColumnsToKeep.append(i)
        mapCols[j] = newIndex
        newIndex += 1
        break
x = x[:,indexColumnsToKeep]

#Remove lines containing NaNs

indexLinesToKeep = []
for ind,line in enumerate(x):
  if np.all(np.logical_not(np.isnan(line))):
    indexLinesToKeep.append(ind)

x = x[indexLinesToKeep, :]
y = y[indexLinesToKeep]

print(x.shape)
#Adjust value to bring them closer to 0 to avoid overflows

x[:, mapCols[0]][np.where(x==1)[0]] = 0

x[:, mapCols[1]][np.where(x==2)[0]] = 0
tmp = []
for ind,line in enumerate(x):
  if line[mapCols[1]] != 7 and line[mapCols[1]] != 9:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

x[:,mapCols[2]][np.where(x==3)[0]] = 0
x[:,mapCols[2]][np.where(x==2)[0]] = 0
tmp = []
for ind,line in enumerate(x):
  if line[mapCols[2]] == 9:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

x[:, mapCols[3]] /= 100

x[:, mapCols[4]][np.where(x==2)[0]] = 0
tmp = []
for ind,line in enumerate(x):
  if line[mapCols[4]] != 7 and line[mapCols[4]] != 9:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

x[:, mapCols[5]][np.where(x==2)[0]] = 0
tmp = []
for ind,line in enumerate(x):
  if line[mapCols[5]] != 7 and line[mapCols[5]] != 9:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

x[:, mapCols[6]][np.where(x==2)[0]] = 0
x[:, mapCols[6]][np.where(x==3)[0]] = 0
x[:, mapCols[6]][np.where(x==1)[0]] = 2
x[:, mapCols[6]][np.where(x==4)[0]] = 1
tmp = []
for ind,line in enumerate(x):
  if line[mapCols[6]] != 7 and line[mapCols[6]] != 9:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

x[:, mapCols[7]][np.where(x==2)[0]] = 0
tmp = []
for ind,line in enumerate(x):
  if line[mapCols[7]] != 9:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

x[:, mapCols[8]][np.where(x==2)[0]] = 0
tmp = []
for ind,line in enumerate(x):
  if line[mapCols[8]] != 9:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

x[:, mapCols[9]][np.where(x==2)[0]] = 0
tmp = []
for ind,line in enumerate(x):
  if line[mapCols[9]] != 9:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

x[:, mapCols[10]][np.where(x==1)[0]] = 0
x[:, mapCols[10]][np.where(x==2)[0]] = 1
tmp = []
for ind,line in enumerate(x):
  if line[mapCols[10]] != 9:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

x[:, mapCols[11]][np.where(x==2)[0]] = 0
tmp = []
for ind,line in enumerate(x):
  if line[mapCols[11]] != 7 and line[mapCols[11]] != 9:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

x[:, mapCols[12]][np.where(x==2)[0]] = 0
tmp = []
for ind,line in enumerate(x):
  if line[mapCols[12]] != 7 and line[mapCols[12]] != 9:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

tmp = []
for ind,line in enumerate(x):
  if line[mapCols[13]] != 7 and line[mapCols[13]] != 9:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

x[:, mapCols[14]][np.where(x==88)[0]] = 0
tmp = []
for ind,line in enumerate(x):
  if line[mapCols[14]] != 77 and line[mapCols[14]] != 99:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

x[:, mapCols[15]][np.where(x==88)[0]] = 0
tmp = []
for ind,line in enumerate(x):
  if line[mapCols[15]] != 77 and line[mapCols[15]] != 99:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

x[:, mapCols[16]][np.where(x==2)[0]] = 0
tmp = []
for ind,line in enumerate(x):
  if line[mapCols[16]] != 7 and line[mapCols[16]] != 9:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

x[:, mapCols[17]][np.where(x==2)[0]] = 0

tmp = []
for ind,line in enumerate(x):
  if line[mapCols[18]] != 14:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

tmp = []
for ind,line in enumerate(x):
  if line[mapCols[19]] != 9:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

tmp = []
for ind,line in enumerate(x):
  if line[mapCols[20]] != 99 and line[mapCols[20]] != 77:
    tmp.append(ind)
x = x[tmp,:]
y = y[tmp]

print(x.shape)

np.savetxt("x_train_processed.csv", x)
np.savetxt("y_train_processed.csv", y)