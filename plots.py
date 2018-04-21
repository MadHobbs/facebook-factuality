# f1 
baseline = 0.775062848734
x = ["Random Forest", "SVC", "Perceptron"]
values = np.array([30., 87.3, 99.9, 3.33, 50.0])
x = range(len(values))

# split it up
above_threshold = np.maximum(values - threshold, 0)
below_threshold = np.minimum(values, threshold)

# and plot it
fig, ax = plt.subplots()
ax.bar(x, below_threshold, 0.35, color="g")
ax.bar(x, above_threshold, 0.35, color="r",
        bottom=below_threshold)

# horizontal line indicating the threshold
ax.plot([0., 4.5], [threshold, threshold], "k--")

fig.savefig("look-ma_a-threshold-plot.png")