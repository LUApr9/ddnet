import matplotlib
import matplotlib
matplotlib.use('Agg')
print(matplotlib.get_backend())
import matplotlib.pyplot as plt

plt.plot([1,2,3])
plt.show()
plt.savefig("test.png")
plt.close()