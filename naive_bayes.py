import numpy as np
import math
from pprint import pprint


def readMatrix(filename):
	'''This function parses the data into a matrix 
	for ready use in implementation of naive bayes algorithm'''
	fid = open(filename)
	headerline = fid.readline()
	rowscols = list(map(int,(fid.readline().split(" "))))
	
	tokenlist = fid.readline().split(" ")
	tokenlist = tokenlist[:-1]
	
	matrix = np.zeros([rowscols[0],rowscols[1]],dtype=int)
	
	category = []
	#print(category)

	for m in range(int(rowscols[0])):
		lines = list(map(int,(fid.readline().split(" "))))
		list1 = [lines[0]]
		category = category+list1
		lines = lines[1:-1]		
		#print(category)
		for i in lines:
			matrix[m][i-1] += 1
			#matrix = matrix[1:]

	#print(np.transpose(matrix))
	return [matrix, tokenlist,category,rowscols]





my_list = readMatrix("MATRIX.TRAIN.1400")
dimensions = my_list[-1]
category = np.asarray(my_list[-2])
#print(category)
Matrix = (np.asarray(my_list[0]))
#print(Matrix)

SP_Matrix = np.array([])
HP_Matrix = np.array([])




for i in range(len(category)):
	if category[i] == 1:
		SP_Matrix = np.append(SP_Matrix, Matrix[i])
	else:
		HP_Matrix = np.append(HP_Matrix, Matrix[i])
SP_Matrix_2D = np.reshape(SP_Matrix, (-1, dimensions[1]))
HP_Matrix_2D = np.reshape(HP_Matrix, (-1, dimensions[1]))
#print(SP_Matrix_2D.shape)
#print(HP_Matrix_2D.shape)

SP_Col_Sum = SP_Matrix_2D.sum(axis=0)
HP_Col_Sum = HP_Matrix_2D.sum(axis=0)
#print(SP_Col_Sum)
#print(HP_Col_Sum)

SP_Prob_X = SP_Col_Sum/SP_Col_Sum.sum()
HP_Prob_X = HP_Col_Sum/HP_Col_Sum.sum()

#print(SP_Matrix_2D.shape)
																					

#print(SP_Matrix.shape())
#print(len(SP_Matrix.sum(axis = 1)))




'''Calculating class probabilities'''
Spam_prob = np.sum(category)/ np.size(category)
Ham_prob = 1- Spam_prob
#print(Spam_prob)
#-print(Ham_prob)


Prob_X = Spam_prob * SP_Prob_X + Ham_prob * HP_Prob_X


'''Testing the data and predicting values'''

fid = open("MATRIX.TEST")
x1 = fid.readline()
x2 = list(map(int, fid.readline().split(" ")))
x3 = fid.readline()
x4 = fid.readlines()
test_category = []
px = []
px_y = []
px_y_np = []
py_x = []
py_x_np = []
test_output = []
print(type(test_category))
data = [[int(n) for n in line.split()] for line in x4]
y = np.array([np.array(xi) for xi in data])
i=0
z=0
for i in range(x2[0]):
	px.append(np.sum(Prob_X[np.unique(y[i])]))
	px_y.append(np.sum(SP_Prob_X[np.unique(y[i])]))
	px_y_np.append(np.sum(HP_Prob_X[np.unique(y[i])]))
	py_x.append(px_y[i] * Spam_prob / px[i])
	py_x_np.append(px_y_np[i] * Ham_prob / px[i])
	test_category.append(y[i][0])
	i = i+1

pprint(list(zip(py_x, py_x_np)))




'''
To find accuracy'''
for j in range(int(x2[0])):
	if py_x > py_x_np:
		test_output.append(1)
	else:
		test_output.append(0)

for k in range(int(x2[0])):
	if test_output[k] == test_category[k]:
		z = z+1
		print(z)
accuracy = z/int(x2[0])
print(accuracy *100)


