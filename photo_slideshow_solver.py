from gurobipy import *
import sys
from gurobipy import GRB
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
import time
import threading
import signal

# Subdomain name: psinstgenerator.erzen.tk

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


    result = None
    def solve_thread(self,solver):
        self.result = solver.Solve()

    def create_GORT_model(self):
        """Google OR Tools model"""
        solver=pywraplp.Solver.CreateSolver('BOP')

        # Decision variables
        z = {}
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    z[(i, j)] = solver.BoolVar(name='z' + str(i)+ ',' + str(j))
        
        # Constraints
        for sp in self.same_photos:
            i = sp[0]
            j = sp[1]
            solver.Add(sum(sum(z[(a, b)] for b in range(self.N) 
                if b != a and ((a == i and b == j) or (a == j or b == i))) 
                for a in range(self.N)) <= 0, name='same photos')
        
        for i in range(self.N ):
            solver.Add(sum(z[(i, j)] for j in range(self.N) if j != i) <= 1,
                        name='one slide is placed after slide i')
        
        for j in range(self.N):
            solver.Add(sum(z[(i, j)] for i in range(self.N) if j != i) <= 1,
                        name='one slide is placed after slide j ')
        
        for i in range(self.N):
            for j in range(self.N):
                if i!=j:
                    solver.Add(z[(i,j)]+z[(j,i)] <= 1, name='no two simetric transitions are allowed')
        
        for k in range(self.H):
            s1=sum(z[(k,i)] for i in range(self.H) if i!=k)
            s2=sum(z[(i,k)] for i in range(self.H) if i!=k)
            solver.Add(s1+s2<= 2, name='each slide is used in at most one transition')
        
        solver.Maximize(sum(sum(z[(i, j)]* self.transition_interest[self.transform_tuple((i, j))] 
                        for j in range(self.N) if j != i) for i in range(self.N)))
        
        # Sets a time limit of 10 seconds.
        # solver.parameters.max_time_in_seconds = 10.0
        # solver.SetTimeLimit(30 * 1000)
        # solver.set_time_limit(10)
        solver.EnableOutput()
        # status = solver.Solve()
        # solver.InterruptSolve()
        result = None
        solve_model = lambda: solver.Solve()
        

        thread = threading.Thread(target= self.solve_thread, args=[solver])
        
        time_limit_seconds = 30*60
        t0 = time.time()
        # solver.Solve()
        # while solver.NumErrors() == 0 and solver.NumFeasiblePoints() < 1 and time.time() - t0 < time_limit_seconds:
        #     solver.NextSolution()

        thread.start()

        while thread.is_alive():
            if time.time() - t0 > time_limit_seconds:
                # Stop the solver if it's taking too long
                solver.InterruptSolve()
                # status = solver.ResultStatus()
                break
            time.sleep(0.1)

        thread.join()

        # statuss = solver.ResultStatus()

        # status = solve_model()
        # solver.objective_callback = 
        # solver.EnableOutput()

        # status = solver.status()

        # timeout = 0.0001
        # handler = lambda signum, frame: print("Time limit exceeded")
        # signal.signal(signal.SIGALRM, handler)
        # signal.alarm(timeout)
        # try:
        #     start_time = time.time()
        #     status = solver.Solve()
        #     elapsed_time = time.time() - start_time
        # except TimeoutError as e:
        #     print(f"Solver timed out after {timeout} seconds")
        #     status = pywraplp.Solver.TIMEOUT
        status = self.result
        if status == pywraplp.Solver.OPTIMAL:
            print('Solution:')
            print('Objective value =', solver.Objective().Value())
        else:
            print('The problem does not have an optimal solution.')  
        
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    if z[(i, j)].solution_value()==1:
                        print(z[(i, j)].name(), ' = ',  z[(i, j)].solution_value())        
    

    


    def stop_function(self):
        if time.time() - self.start_time >= self.time_limit:
            return True
        else:
            return False
    

    def create_model(self):
        m = gurobipy.Model('photo slide show')
        m.setParam('WorkLimit', 3*10)
        # Decision variables
        z = {}
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    z[(i, j)] = m.addVar(vtype=GRB.BINARY, name='z' + str(i)+ ',' + str(j))

        # Constraints
        for sp in self.same_photos:
            i = sp[0]
            j = sp[1]
            m.addConstr(quicksum(quicksum(z[(a, b)] for b in range(self.N) 
                if b != a and ((a == i and b == j) or (a == j or b == i))) 
                for a in range(self.N)) <= 0, name='same photos')

        for i in range(self.N ):
            m.addConstr(quicksum(z[(i, j)] for j in range(self.N) if j != i) <= 1,
                        name='one slide is placed after slide i')

        for j in range(self.N):
            m.addConstr(quicksum(z[(i, j)] for i in range(self.N) if j != i) <= 1,
                        name='one slide is placed after slide j ')
        
        for i in range(self.N):
            for j in range(self.N):
                if i!=j:
                    m.addConstr(z[(i,j)]+z[(j,i)] <= 1, name='no two simetric transitions are allowed')

        for k in range(self.H):
            s1=quicksum(z[(k,i)] for i in range(self.H) if i!=k)
            s2=quicksum(z[(i,k)] for i in range(self.H) if i!=k)
            m.addConstr(s1+s2<= 2, name='each slide is used in at most one transition')
        
        m.update()
        m.Params.timeLimit = 20.0
        m.setObjective(quicksum(quicksum(z[(i, j)]* self.transition_interest[self.transform_tuple((i, j))] 
                        for j in range(self.N) if j != i) for i in range(self.N)), sense=GRB.MAXIMIZE)

        return m
    
    
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

def help_function():
    """Define help function"""
    print("The solver should be callde usign the command 'python photo_slideshow_solver.py instance_name.txt'")
    print("Example: python photo_slideshow_solver.py c_memorable_moments_50.txt")

if __name__=="__main__":
    """ Create an instance of the model and solve the problem"""
    arguments=sys.argv
    if len(arguments)!=2:
        help_function()
        exit()
    else:
        file_name = arguments[1]
        ps=PhotoSlideShow(file_name)
        
        # Gurobi model
        # model=ps.create_model()
        # objective_value=ps.solve_problem(model)
        # print("Objective value:: "+str(objective_value))
        

        # Googl OR Tools
        model=ps.create_GORT_model()


    # file_name = "P50_H0_V50.txt"
    # ps=PhotoSlideShow(file_name)
    # model=ps.create_model()
    # objective_value=ps.solve_problem(model)
    # print("Solution quality: "+str(objective_value))

