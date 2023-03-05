import copy
import sys

def initializeInputForm(fileName):
	inputForm = []
	file = open(fileName, 'r')
	totalPhoto = int(file.readline())
	fileRows = filter(None,file.read().split('\n'))
	for row in fileRows:
		inputForm.append(row.split(' '))
	return totalPhoto,inputForm

def getTotalDifferentTags(photoTags1,photoTags2):
	differentTags = list(set(photoTags1) - set(photoTags2))
	return len(differentTags)

def initializeSolution(fileName):
	solution = []
	file = open(fileName, 'r')
	nrOfRows = int(file.readline())
	fileRows = filter(None,file.read().split('\n'))
	for row in fileRows:
		i = row.split(' ')
		if len(i) == 1:
			solution.append(int(i[0]))
		else:
			solution.append([int(i[0]),int(i[1])])
	return nrOfRows,solution

def combineTags(photoTag1,photoTag2):
  	return list(set(photoTag1) | set(photoTag2))

def getPhotoTags(position,inputForm):
	if isinstance(position, int):
		tmp = copy.copy(inputForm[position])
		tmp[0:2] = []
		return tmp
	else:
		tmp_1 = copy.copy(inputForm[position[0]])
		tmp_1[0:2] = []
		tmp_2 = copy.copy(inputForm[position[1]])
		tmp_2[0:2] = []
		return combineTags(tmp_1,tmp_2)

def getMinimumBetweenTwoPhotos(photo1,photo2):
	countTagsOnlyPhoto1 = getTotalDifferentTags(photo1, photo2)
	countTagsOnlyPhoto2 = getTotalDifferentTags(photo2, photo1)
	countSameTags = len(photo1) - countTagsOnlyPhoto1
	return min(countTagsOnlyPhoto1, countTagsOnlyPhoto2, countSameTags)

def calculateInitialFitnes(solution,inputForm):
	initialFitness = 0
	for i in range(0,len(solution)-1,1):
		initialFitness = initialFitness + getMinimumBetweenTwoPhotos(getPhotoTags(solution[i],inputForm),getPhotoTags(solution[i+1],inputForm))
	return initialFitness

def CheckIfSolutionCompleteHardConstrains(solution,inputForm):
	tmpArray = []
	for i in range(0,len(solution),1):
		if isinstance(solution[i],int):
			tmpArray.append(solution[i])
			if not isHorizontal(solution[i],inputForm):
				print("Error! Gabim ne foton Horizontale");
				print("Detajet:")
				print("Gabimi ndodhet ne rreshtin: "+ str(i));
				print(inputForm[solution[i]])
			
		else:
			tmpArray.append(solution[i][0])
			tmpArray.append(solution[i][1])
			if not isVertical(solution[i][0],solution[i][1],inputForm):
				print("Error! Gabim ne foton Vertikale \n");
				print("Detajet:")
				print("Gabimi ndodhet ne rreshtin: "+ str(i));
				print(inputForm[solution[i][0]])
				print(inputForm[solution[i][1]])
	tmp = IsValidSolution(tmpArray)
	if len(tmp) > 0:
		print("Error! Nje fotografi eshte perdorur me shume se dy here: ")
		print("Detajet:")
		print(tmp)

def isHorizontal(photo,inputForm):
	if inputForm[photo][0] == 'H':
		return True
	return False

def isVertical(photo1,photo2,inputForm):
	if inputForm[photo1][0] == 'V' and inputForm[photo2][0] == 'V':
		return True
	return False

def IsValidSolution(solution):
	return list(set([x for x in solution if solution.count(x) > 1]))		

def helpFunction():
	print("The validator should be called as in the follwing: python validator.py [instnace name] [solution name]")
	print("Example: python validator.py c_memorable_moments_20.txt c_memorable_moments_20.txt_solution_6.0.txt")
	exit()

if __name__ == "__main__":
	arguments = sys.argv
	if len(arguments) != 3:
		helpFunction()
	input_file_name ="Instances/"+arguments[1]
	if input_file_name == None:
		helpFunction()
	solution_file_name ="Solutions/"+ arguments[2]

	totalPhoto,inputForm = initializeInputForm(input_file_name)
	numberOfRows,solution = initializeSolution(solution_file_name)
	print("==  Calculating fitness ... ==")
	print(calculateInitialFitnes(solution,inputForm))
	print("==  Validating solution ... ==")
	CheckIfSolutionCompleteHardConstrains(solution,inputForm)