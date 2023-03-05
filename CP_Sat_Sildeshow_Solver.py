from ortools.sat.python import cp_model
import time
import sys
from gurobipy import *
import sys
from gurobipy import GRB
from google.protobuf import text_format
import os

class PhotoSlideShow:
    def __init__(self,file_name):
        self.photos = self.read_instance_from_file(file_name)
        self.M=len(self.photos)
        self.horizontal_photos,self.vertical_photos=self.countHorizontalVertical()
        self.H=len(self.horizontal_photos)
        self.V=len(self.vertical_photos)
        self.NH=self.H
        self.NV=int(self.V*(self.V-1)/2)
        self.N=self.NH+self.NV
        self.possible_slides=self.getPossibleSlides()
        self.same_photos=self.getSamePhotos()
        self.transition, self.transition_interest=self.calculateTransitionInterest()
        self.time_limit = 1
        self.start_time = time.time()
    
    def read_instance_from_file(self,file_name: str):
        result=dict()
        try:
            with open('Instances\\' + file_name, 'r') as f:
                P = int(f.readline())
                photos = {}
                for i in range(P):
                    photo_text = f.readline()
                    photo_data = photo_text.split()
                    photos[i] = photo_data
                result = photos
                f.close()
        except FileNotFoundError as e:
            print(e.strerror)
        except Exception as e:
            print(e.values)
        else:
            pass
        return result

    def countHorizontalVertical(self):
        """Count horizontal and vertical phots"""
        horizontal_photos=dict()
        vertical_photos=dict()
        for id, photos in self.photos.items():
            if photos[0]=='H':
                horizontal_photos[id]=photos[2:]
            else:
                vertical_photos[id]=photos[2:]
        return horizontal_photos,vertical_photos
    
    def getPossibleSlides(self)->dict:
        """Define possible list of slides by considering horizontal and vertical photos"""
        result=dict()
        slide_index=0
        # Get possible slides from Horizontal photos
        for horizontal_photo_id in self.horizontal_photos.keys():
            result[slide_index]=[horizontal_photo_id]
            slide_index+=1
        
        # Get possible slides from Vertical photos
        vertical_photo_ids=[key for key in self.vertical_photos.keys()]
        for i in range(len(vertical_photo_ids)-1):
            for j in range (i+1,len(vertical_photo_ids)):
                first_photo_id=vertical_photo_ids[i]
                second_photo_id=vertical_photo_ids[j]
                result[slide_index]=[first_photo_id,second_photo_id]
                slide_index+=1
        return result

    def getSamePhotos(self)->tuplelist:
        """Get slides that have same photos"""
        result_list=list()
        possible_slide_list=[key for key in self.possible_slides.keys()]
        for i in range(self.H,len(possible_slide_list)-1):
            for j in range(i+1,len(possible_slide_list)):
                slide_1_photos=set(self.possible_slides[i])
                slide_2_photos=set(self.possible_slides[j])
                if len(slide_1_photos.intersection(slide_2_photos))>0:
                    result_list.append((i,j))
        return tuplelist(result_list)

    def calculateTransitionInterest(self)->multidict:
        """Calcualte Transition Interest between all possible slides"""
        result_dict=dict()
        possible_slide_list=[key for key in self.possible_slides.keys()]
        for i in range(0,len(possible_slide_list)-1):
            for j in range(i+1,len(possible_slide_list)):
                slide_1_photos=self.possible_slides[possible_slide_list[i]]
                slide_2_photos=self.possible_slides[possible_slide_list[j]]
                slide_1_tags=set()
                for p in slide_1_photos:
                    for t in self.photos[p][2:]:
                        slide_1_tags.add(t)                
                slide_2_tags=set()
                for p in slide_2_photos:
                    for t in self.photos[p][2:]:
                        slide_2_tags.add(t)
                intersaction12=slide_1_tags.intersection(slide_2_tags)
                difference12=slide_1_tags.difference(slide_2_tags)
                difference21=slide_2_tags.difference(slide_1_tags)
                transition_interest=min(len(intersaction12),len(difference12),len(difference21))
                result_dict[(i,j)]=transition_interest
        return multidict(result_dict)


    def transform_tuple(self, t:tuple)->tuple:
        """Transform tuple indeces"""
        min_index=min(t)
        max_index=max(t)
        return (min_index,max_index)
    

    
    
    def solve_problem(self,m):
        m.optimize()
        m.printAttr('X')
        
        objective_value=int(m.objVal)
        return objective_value

    
    def order_slide_transitions(self,slide_pair_queue:list):
        i=0
        while i <len(slide_pair_queue):
            current_slide_transition=slide_pair_queue[i%len(slide_pair_queue)]
            j=i+1
            restart=False
            while j<len(slide_pair_queue):
                next_slide_transition=slide_pair_queue[j%len(slide_pair_queue)]
                if current_slide_transition[1]==next_slide_transition[0] and j!=i+1:
                    slide_pair_queue.pop(i%len(slide_pair_queue))
                    slide_pair_queue.insert(j-1,current_slide_transition)
                    restart=True
                    break
                elif current_slide_transition[0]==next_slide_transition[1]:
                    slide_pair_queue.pop(j%len(slide_pair_queue))
                    slide_pair_queue.insert(i,next_slide_transition)
                    restart=True
                    break
                j+=1
            if not restart:
                i+=1
            else:
                i=0
        return
    
    def save_solution_to_file(self,slide_list:list,objective_value:float):
        solution_text=str(len(slide_list))+'\n'
        for s in slide_list:
            current_slide_photos=self.possible_slides[s]
            if len(current_slide_photos)==1:
                solution_text=solution_text+str(current_slide_photos[0])+'\n'
            else:
                solution_text=solution_text+str(current_slide_photos[0])+' '+str(current_slide_photos[1])+'\n'
        print("Objective value: {}".format(objective_value))
        print("Num slides: {}".format(len(slide_list)))
        print(slide_list)
        print(solution_text)
        output_file_name=file_name[0:len(file_name)-4]+'_solution_'+str(objective_value)+'.txt'
        with open("Solutions\\"+output_file_name,'w') as f:
            f.write(solution_text)


    def create_GORT_model(self, file_name: str):
        """Google OR Tools model"""

        model = cp_model.CpModel()
        
        
        # Decision variables
        z = {}
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    z[(i, j)] = model.NewBoolVar(name='z' + str(i)+ ',' + str(j))
        
        

        
        for sp in self.same_photos:
            i = sp[0]
            j = sp[1]
            model.Add(sum(sum(z[(a, b)] for b in range(self.N) 
                if b != a and ((a == i and b == j) or (a == j or b == i))) 
                for a in range(self.N)) <= 0)
            
        
        
        for i in range(self.N):
            # First loop
            row_sum = sum(z[(i, j)] for j in range(self.N) if j != i)
            model.Add(row_sum <= 1)

            # Second loop
            col_sum = sum(z[(j, i)] for j in range(self.N) if j != i)
            model.Add(col_sum <= 1)

            # Third loop
            for j in range(self.N):
                if i != j:
                    model.Add(z[(i, j)] + z[(j, i)] <= 1)
        
        for k in range(self.H):
            s1=sum(z[(k,i)] for i in range(self.H) if i!=k)
            s2=sum(z[(i,k)] for i in range(self.H) if i!=k)
            model.Add(s1+s2<= 2)

        model.Maximize(sum(sum(z[(i, j)]* self.transition_interest[self.transform_tuple((i, j))] 
                        for j in range(self.N) if j != i) for i in range(self.N)))
        
        solver = cp_model.CpSolver()
        
        model_proto = model.Proto()
        # model.ExportToFile("Model_30_Photos.proto")
        # x = model_proto.SerializeToString()
        # model_proto.ParseFromString(x)
        # with open('Model_100_Photos.proto', 'wb') as f: 
        #     f.write(model_proto.SerializeToString())

        with open(file_name + '.proto', 'wb') as f: 
            f.write(model_proto.SerializeToString())

        solver.parameters.max_time_in_seconds = 1800
        
        
        status = solver.Solve(model)
        
        toSave = ''

        print(solver.ResponseStats())
        toSave += solver.ResponseStats() + "\n\n"
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print('Solution:')
            print('Objective value = ', solver.ObjectiveValue())
            toSave += 'Objective value = '+ str(solver.ObjectiveValue()) + '\n\n'

        else:
            print('The problem does not have an optimal nor feasible solution.')  
        
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    if solver.Value(z[(i, j)])==1:
                        print('z' + str(i)+ ',' + str(j), ' = ',  solver.Value(z[(i, j)]))   
                        toSave += 'z' + str(i)+ ',' + str(j)+ ' = '+  str(solver.Value(z[(i, j)])) + '\n'
        
        with open('Result for ' + file_name + '.txt', 'w') as f: 
            f.write(toSave)


   

def help_function():
    """Define help function"""
    print("The solver should be callde usign the command 'python photo_slideshow_solver.py instance_name.txt'")
    print("Example: python photo_slideshow_solver.py c_memorable_moments_50.txt")


def solveFromProto(file):
    print("Solving from proto file")
    with open(file, 'rb') as f:
        proto_data = f.read()
    print("File read")
    # pdata = proto_data

    # with open('model.proto', 'r') as f:
    #     model_proto = cp_model_pb2.CpModelProto()
    #     text_format.Parse(f.read(), model_proto)

    
    model = cp_model.CpModel()
    model_proto = model.Proto()
    model_proto.ParseFromString(proto_data)
    print("Proto parsed")
    
    # model.CopyFrom(model_proto)
    # x = text_format.Parse(proto_data, model)
    # model.CopyFrom(x)
    
    
    # for i in range(self.N):
    #     for j in range(self.N):
    #         if i != j:
    #             # z[(i, j)] = model.NewBoolVar(name='z' + str(i)+ ',' + str(j))
    #             model.GetBoolVarFromProtoIndex(i)

    z={}
    # test = model_proto.variables
    for i in range(len(model_proto.variables)):
        z[model_proto.variables[i].name] = model.GetBoolVarFromProtoIndex(i)
    print("Variables created")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1800
    solver.parameters.max_memory_in_mb = 10000

    print("Starting solver!")
    status = solver.Solve(model)
    print("Solved")
    print(solver.ResponseStats())
    print(solver.SolutionInfo())
    print(model.ModelStats())
    
    # t = model_proto.variables[10]
    # x = vars(t)
    # print(x.keys()[0])
    # print(solver.Value(x.values[x.keys()[0]]))
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('Solution:')
        print('Objective value =', solver.ObjectiveValue())
    else:
        print('The problem does not have an optimal nor feasible solution.')  
    


    for i in range(len(model_proto.variables)):
                if solver.Value(z[model_proto.variables[i].name])==1:
                    print(list(z.keys())[i] + ' = ',  solver.Value(z[model_proto.variables[i].name]))
    # for var in model_proto.variables:
    #     print(f'{var}: {solver.Value(var)}')
    # for i,var in enumerate(model_proto.variables):
    #     print(f'{var}: {solver.Value(model_proto.variables[i])}')
    # for i in range(self.N):
    #     for j in range(self.N):
    #         if i != j:
    #             if solver.Value(z[(i, j)])==1:
    #                 print('z' + str(i)+ ',' + str(j), ' = ',  solver.Value(z[(i, j)]))   
    
    return model

if __name__=="__main__":
    """ Create an instance of the model and solve the problem"""
    arguments=sys.argv
    if len(arguments)!=2:
        help_function()
        exit()
    else:
        file_name = arguments[1]
        # toSave = "Hello World!"
        # with open('Results for 100 photos.txt', 'w') as f: 
        #     f.write(toSave)

        # ps=PhotoSlideShow(file_name)
        
        # Gurobi model
        # model=ps.create_model()
        # objective_value=ps.solve_problem(model)
        # print("Objective value:: "+str(objective_value))
        

        # Googl OR Tools
        # model=ps.create_GORT_model()
        # model=ps.create_GORT_model(os.path.splitext(file_name)[0])
        model = solveFromProto("./P5k_H5k_V0.proto")




# model = cp_model.CpModel()