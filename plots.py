import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

data = pd.read_csv('clear.csv')

# likes
print "likes"
print data.groupby(['Rating'])['num_likes'].sum()
# shares
print "shares"
print data.groupby(['Rating'])['num_shares'].sum()
print "likes"
print data.groupby(['Rating'])['num_likes'].sum()
print "loves"
print data.groupby(['Rating'])['num_loves'].sum()
print "wows"
print data.groupby(['Rating'])['num_wows'].sum()
print "hahas"
print data.groupby(['Rating'])['num_hahas'].sum()
print "sads"
print data.groupby(['Rating'])['num_sads'].sum()
print "angrys"
print data.groupby(['Rating'])['num_angrys'].sum()

names = ["likes", "loves", "wows", "hahas", "sads", "angrys"]

mixture = [1087910, 75579, 51525, 76839, 33982, 173354]
false = [293394, 10869, 15450, 44956, 6785, 45151]
true = [3287519, 246986, 101322, 252750, 102979, 363896]

names = ["mixture of true and false", "mostly false", "mostly true"]
likes = [1087910, 293394, 3287519]
total = float(sum(likes))
blueBars = [likes[0]/total, likes[1]/total, likes[2]/total]

loves = [75579, 10869, 246986]
total = float(sum(loves))
pinkBars = [loves[0]/total, loves[1]/total, loves[2]/total]

wows = [51525, 15450, 101322]
total = float(sum(wows))
orangeBars = [wows[0]/total, wows[1]/total, wows[2]/total]

hahas = [76839, 44956, 252750]
total = float(sum(hahas))
yellowBars = [hahas[0]/total, hahas[1]/total, hahas[2]/total]

sads = [33982, 6785, 102979]
total = float(sum(sads))
purpleBars = [sads[0]/total, sads[1]/total, sads[2]/total]

angrys = [173354, 45151, 363896]
total = float(sum(angrys))
redBars = [angrys[0]/total, angrys[1]/total, angrys[2]/total]


raw_data = {'likes': likes, 'loves': loves, 'wows': wows, 'hahas': hahas, \
'sads': sads, 'angrys': angrys}
df = pd.DataFrame(raw_data)

totals = [i+j+k+l+m+n for i,j,k,l,m,n in zip(df['likes'], df['loves'], df['wows'], df["hahas"], df["sads"], df["angrys"])]
blueBars = [i / float(j) * 100 for i,j in zip(df['likes'], totals)]
pinkBars = [i / float(j) * 100 for i,j in zip(df['loves'], totals)]
orangeBars = [i / float(j) * 100 for i,j in zip(df['wows'], totals)]
yellowBars = [i / float(j) * 100 for i,j in zip(df['hahas'], totals)]
purpleBars = [i / float(j) * 100 for i,j in zip(df['sads'], totals)]
redBars = [i / float(j) * 100 for i,j in zip(df['angrys'], totals)]

barWidth = 0.85
r = range(len(names))
# Create green Bars
plt.bar(r, blueBars, color='blue', edgecolor='white', width=barWidth, label = "likes")
plt.bar(r, pinkBars, bottom=blueBars, color='pink', edgecolor='white', width=barWidth, label = "loves")
plt.bar(r, orangeBars, bottom=[i+j for i,j in zip(blueBars, pinkBars)], color='orange', edgecolor='white', width=barWidth, label = "wows")
plt.bar(r, yellowBars, bottom=[i+j+k for i,j,k in zip(blueBars, pinkBars, orangeBars)], color='yellow', edgecolor='white', width=barWidth, label = "hahas")
plt.bar(r, purpleBars, bottom=[i+j+k+l for i,j,k,l in zip(blueBars, pinkBars, orangeBars, yellowBars)], color='purple', edgecolor='white', width=barWidth, label = "sads")
plt.bar(r, redBars, bottom=[i+j+k+l+m for i,j,k,l,m in zip(blueBars, pinkBars, orangeBars, yellowBars, purpleBars)], color='red', edgecolor='white', width=barWidth, label = "angrys")
 
# Custom x axis
plt.xticks(r, names)
plt.xlabel("Factuality Rating")
plt.legend(loc = 4)
plt.ylabel("Percentage")
# Show graphic
plt.show()

#######################

mixture = [65, 8, 142]
false = [19, 0, 56]
true = [252, 1073, 262]

raw_data = {'mixture': mixture, 'mostly false': false,'mostly true': true}
df = pd.DataFrame(raw_data)

r = [0,1,2]
totals = [i+j+k for i,j,k in zip(df['mixture'], df['mostly false'], df['mostly true'])]
blueBars = [i / float(j) * 100 for i,j in zip(df['mixture'], totals)]
orangeBars = [i / float(j) * 100 for i,j in zip(df['mostly false'], totals)]
greenBars = [i / float(j) * 100 for i,j in zip(df['mostly true'], totals)]

barWidth = 0.85
names = ('Left','Mainstream','Center')
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label = "mostly true")
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, label = "mostly false")
# Create blue Bars
plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', width=barWidth, label = "mixture of true and false")
 
# Custom x axis
plt.xticks(r, names)
plt.xlabel("News Category")
plt.legend()
plt.ylabel("Percentage")
# Show graphic
plt.show()


'''fig, ax = plt.subplots()
# left
"print left"
print data.groupby(['Rating'])['Category_left'].sum()
labels = ["mixture true and false", "mostly false", "mostly true"]
left = [65, 19, 252]
# mainstream
print "mainstream"
print data.groupby(['Rating'])['Category_mainstream'].sum()
mainstream = [8, 0, 1073]
# right
print "rights"
print data.groupby(['Rating'])['Category_right'].sum()
right = [142, 56, 262]


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

## Absolute count

ax1.bar(range(len(data1)), mixture, label='mix', alpha=0.5, color='b')
ax1.bar(range(len(data2)), false, bottom=data1, label='data 2', alpha=0.5, color='r')
ax1.bar(range(len(data2)), true, bottom=data1, label='data 2', alpha=0.5, color='r')
plt.sca(ax1)
plt.xticks([0.4, 1.4, 2.4],  ['category 1', 'category 2', 'category 3'])
ax1.set_ylabel("Count")
ax1.set_xlabel("")
plt.legend(loc='upper left')
'''
# f1 
baseline = 0.846
x = ["Random Forest"]
values = np.array([0.896])
x = range(len(values))

# split it up
#above_threshold = np.maximum(values - threshold, 0)
#below_threshold = np.minimum(values, threshold)

# and plot it
fig, ax = plt.subplots()
ax.bar(x, values, color="g", label = "Random Forest")
#ax.bar(x, above_threshold, 0.35, color="r",
        #bottom=below_threshold)

# horizontal line indicating the threshold
ax.plot([0., 4.5], [baseline, baseline], "k--", label = "Majority Vote")
plt.title("Held-out Accuracy Across Different Models")
plt.ylabel("Accuracy")
plt.xlabel("Classifier")
plt.legend()
plt.show()
