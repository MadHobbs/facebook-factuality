import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc

data = pd.read_csv('clear.csv')
'''
feats = ["Category_mainstream", "num_shares", "Category_right", "num_wows", "num_likes", "num_reactions", \
"num_comments", "num_angrys", "num_hahas", "num_sads", "num_loves", "donald", \
"trump", "Category_left", "clinton", "president", "debate", "says", "video", \
"republican", "said", "america", "george", "americans", "racist"]

imps = [0.135356, 0.099026, 0.098160, 0.066903, 0.064147, 0.061870, 0.061593, 0.050732, \
0.047292, 0.045151, 0.044853, 0.019866, \
0.015771, 0.013685, 0.008299, 0.005285, 0.004969, 0.004541, 0.004141, 0.004071, \
0.003881, 0.003496, 0.003278, 0.003093, 0.003005]

r = range(len(feats))
plt.bar(r, imps, color = "blue")
plt.xticks(r, feats, rotation = 70)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Random Forests Feature Importance")
# Show graphic
plt.show()

# likes
print "likes"
#print data.groupby(['Rating'])['num_likes'].sum()
# shares
print "shares"
print data.groupby(['Rating'])['num_shares'].mean()
print "comments"
print data.groupby(['Rating'])['num_comments'].mean()
print "reactions"
print data.groupby(['Rating'])['num_reactions'].mean()
print "likes"
print data.groupby(['Rating'])['num_likes'].mean()
print "loves"
print data.groupby(['Rating'])['num_loves'].mean()
print "wows"
print data.groupby(['Rating'])['num_wows'].mean()
print "hahas"
print data.groupby(['Rating'])['num_hahas'].mean()
print "sads"
print data.groupby(['Rating'])['num_sads'].mean()
print "angrys"
print data.groupby(['Rating'])['num_angrys'].mean()

names = ["mostly true", "mixture of true and false", "mostly false"]
r = range(len(names))
median_shares = [49, 614, 1031]
median_comments = [111, 142, 138]
median_reactions = [357, 1404, 1348]
# Custom x axis
plt.bar(r, median_shares, color = "blue", label = "shares")
plt.bar(r, median_comments, bottom = median_shares, color = "grey", label = "comments")
plt.bar(r, median_reactions, bottom = [i+j for i,j in zip(median_shares, median_comments)], color = "red", label = "reactions")
plt.xticks(r, names)
plt.xlabel("Factuality Rating")
plt.legend(loc = 4)
plt.ylabel("Median Count per Post")
# Show graphic
plt.show()


names = ["likes", "loves", "wows", "hahas", "sads", "angrys"]

mixture = [1087910, 75579, 51525, 76839, 33982, 173354]
false = [293394, 10869, 15450, 44956, 6785, 45151]
true = [3287519, 246986, 101322, 252750, 102979, 363896]

names = ["mixture of true and false", "mostly false", "mostly true"]
likes_m = [880, 832, 219]
likes = [5060.046512, 3911.920000, 2071.530561]
total = float(sum(likes))
blueBars = [likes[0]/total, likes[1]/total, likes[2]/total]

loves_m = [13, 8, 11]
loves = [351.530233, 144.920000, 155.630750]
total = float(sum(loves))
pinkBars = [loves[0]/total, loves[1]/total, loves[2]/total]

wows_m = [34, 49, 5]
wows = [239.651163, 206.000000, 63.844991]
total = float(sum(wows))
orangeBars = [wows[0]/total, wows[1]/total, wows[2]/total]

hahas_m = [35, 45, 16]
hahas = [357.390698, 599.413333, 159.262760]
total = float(sum(hahas))
yellowBars = [hahas[0]/total, hahas[1]/total, hahas[2]/total]

sads_m = [10, 11, 2]
sads = [158.055814, 90.466667, 64.889099]
total = float(sum(sads))
purpleBars = [sads[0]/total, sads[1]/total, sads[2]/total]

angrys_m = [51, 120, 14]
angrys = [806.297674, 602.013333, 229.298047]
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
plt.bar(r, likes, color='blue', edgecolor='white', width=barWidth, label = "likes")
plt.bar(r, loves, color='pink', edgecolor='white', width=barWidth, label = "loves")
plt.bar(r, wows, color='orange', edgecolor='white', width=barWidth, label = "wows")
plt.bar(r, hahas, color='yellow', edgecolor='white', width=barWidth, label = "hahas")
plt.bar(r, sads,  color='purple', edgecolor='white', width=barWidth, label = "sads")
plt.bar(r, angrys, color='red', edgecolor='white', width=barWidth, label = "angrys")
 
# Custom x axis
plt.xticks(r, names)
plt.xlabel("Factuality Rating")
plt.legend()
plt.ylabel("Mean Count per Post")
# Show graphic
plt.show()

#######################
'''
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
plt.bar(r, raw_data["mostly true"], color='#b5ffb9', edgecolor='white', width=barWidth, label = "mostly true")
# Create orange Bars
plt.bar(r, raw_data["mixture"], bottom=raw_data["mostly true"], color='#f9bc86', edgecolor='white', width=barWidth, label = "mixture of true and false")
# Create blue Bars
plt.bar(r, raw_data["mostly false"], bottom=[i+j for i,j in zip(raw_data["mostly true"], raw_data["mixture"])], color='#a3acff', edgecolor='white', width=barWidth, label = "mostly false")
 
# Custom x axis
plt.xticks(r, names)
plt.xlabel("News Category")
plt.legend()
plt.ylabel("Percentage")
# Show graphic
plt.show()

'''
fig, ax = plt.subplots()
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
names = ["RF Meta Only", "Random Forest", "Linear SVC", "Perceptron"]
values = np.array([0.872, 0.868, 0.853, 0.842])
x = range(len(values))

# split it up
#above_threshold = np.maximum(values - threshold, 0)
#below_threshold = np.minimum(values, threshold)

# and plot it
fig, ax = plt.subplots()
ax.bar(x, values, color="g")
#ax.bar(x, above_threshold, 0.35, color="r",
        #bottom=below_threshold)

plt.xticks(x, names, rotation = 70)

# horizontal line indicating the threshold
ax.plot([0., 4.5], [baseline, baseline], "k--", label = "Majority Vote")
plt.title("Held-out F1 Score Across Different Models")
plt.ylabel("F1 Score")
plt.xlabel("Classifier")
plt.legend()
plt.ylim(0,1)
plt.show()
