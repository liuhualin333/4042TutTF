import matplotlib.pyplot as plt
classA=[[5,1], [7,3], [3,2], [5,4]]
classB=[[0,0], [-1,-3], [-2,3], [-3,0]]
for point in classA:
	plt.plot(point[0], point[1], 'ro')
for point in classB:
	plt.plot(point[0], point[1], 'bo')
plt.axis([0, 10, 0, 10])
plt.show()