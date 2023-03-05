import os
from ortools.sat.python import cp_model
import time
import sys
from gurobipy import *
import sys
from gurobipy import GRB
from google.protobuf import text_format
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import multiprocessing as mp
import itertools
import json

lock = mp.Lock()

# def add_constraints(args):
#     z, same_photos_chunk, N = args

#     constraints = []
#     for sp in same_photos_chunk:
#         i = sp[0]
#         j = sp[1]
#         constraint_expr = sum(sum(z[(a, b)] for b in range(N) 
#             if b != a and ((a == i and b == j) or (a == j or b == i))) 
#             for a in range(N))
#         constraints.append({'constraint': str(constraint_expr), 'operator': '<=', 'value': '0'})
#     return constraints

def  process_chunk(z, q, chunk, N, counter):
    result = []
    print(f"Processing chunk {counter}...")
    constraints = []
    for sp in chunk:
        i = sp[0]
        j = sp[1]
        constraint_expr = sum(sum(z[(a, b)] for b in range(N) 
            if b != a and ((a == i and b == j) or (a == j or b == i))) 
            for a in range(N))
        
        constraints.append(f"{str(constraint_expr)} <= 0")

    print(f"Chunk {counter} has been processed. Writing to file...")
    
    with open(f"./constraint_chunks/constraints_{counter}.txt", "w") as f:
        f.write("\n".join(constraints))
    q.put(counter)


def add_constraints(args):
    z, same_photos_chunk, N, file_suffix, q = args
    print(f"Processing chunk {file_suffix}")
    constraints = []
    for sp in same_photos_chunk:
        i = sp[0]
        j = sp[1]
        constraint_expr = sum(sum(z[(a, b)] for b in range(N) 
            if b != a and ((a == i and b == j) or (a == j or b == i))) 
            for a in range(N))
        constraints.append(f"{str(constraint_expr)} <= 0")
    
    # Write constraints to file
    print(f"Chunk {file_suffix} has been processed. Writing to file...")
    with open(f"./constraint_chunks/constraints_{file_suffix}.txt", "w") as f:
        f.write("\n".join(constraints))

    q.put(file_suffix)


# def add_constraints(args):
#     z, same_photos_chunk, N = args

#     constraints = []
#     for sp in same_photos_chunk:
#         i = sp[0]
#         j = sp[1]
#         constraints.append(sum(sum(z[(a, b)] for b in range(N) 
#             if b != a and ((a == i and b == j) or (a == j or b == i))) 
#             for a in range(N)) <= 0)
#         # constraints.append(constraint_expr <= 0)
#     return list(constraints)

# def add_constraints(args):
#     z, same_photos_chunk, N = args

#     for sp in same_photos_chunk:
#         i = sp[0]
#         j = sp[1]
#         yield sum(sum(z[(a, b)] for b in range(N) 
#             if b != a and ((a == i and b == j) or (a == j or b == i))) 
#             for a in range(N)) <= 0



def create_constraints(chunk, N, z, results_queue: mp.Queue):
    # model = cp_model.CpModel()

    # z = {}
    # for i in range(N):
    #     for j in range(N):
    #         if i != j:
    #             z[(i, j)] = model.NewBoolVar(f'z_{i}_{j}')

    constraints = []
    for sp in chunk:
        i = sp[0]
        j = sp[1]
        constraints.append(sum(sum(z[(a, b)] for b in range(N) 
            if b != a and ((a == i and b == j) or (a == j or b == i))) 
            for a in range(N)) <= 0)

    
    results_queue.put(constraints)
    # return constraints


# def add_constraints(z, chunk, N, model):
#     # Add constraints to the model
    
#     # model = cp_model.CpModel()
#     lock.acquire()
#     try:

#         for sp in chunk:
#             i = sp[0]
#             j = sp[1]
#             model.Add(sum(sum(z[(a, b)] for b in range(N) if b != a and ((a == i and b == j) or (a == j or b == i))) for a in range(N)) <= 0)
#     finally:
#         lock.release()

# lock = mp.Lock()
# def add_constraints_process(chunk, z, model,N):
#     lock.acquire()
#     try:
#         for sp in chunk:
#             i = sp[0]
#             j = sp[1]
#             model.Add(sum(sum(z[(a, b)] for b in range(N) 
#                 if b != a and ((a == i and b == j) or (a == j or b == i))) 
#                 for a in range(N)) <= 0)
#     finally:
#         lock.release()

# def add_constraints(chunk, z, model_dict, lock, N):
#     with lock:
#         model = model_dict["model"]
#         for sp in chunk:
#             i = sp[0]
#             j = sp[1]
#             model.Add(sum(sum(z[(a, b)] for b in range(N) 
#                 if b != a and ((a == i and b == j) or (a == j or b == i))) 
#                 for a in range(N)) <= 0)


# def add_constraints(model, z, chunk, N, lock):
#     # Add constraints to the model
#     lock.acquire()
#     try:
#         for sp in chunk:
#             i = sp[0]
#             j = sp[1]
#             model.Add(sum(sum(z[(a, b)] for b in range(N) if b != a and ((a == i and b == j) or (a == j and b == i))) for a in range(N)) <= 0)
#     finally:
#         lock.release()


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


    def add_constraint(self, sp, z, N, model):
        i, j = sp
        model.Add(sum(sum(z[(a, b)] for b in range(N) if b != a and ((a == i and b == j) or (a == j or b == i))) for a in range(N)) <= 0)


    # def add_constraints(self, chunk:list, z: dict, model: cp_model.CpModel):
    #     for sp in chunk:
    #         i = sp[0]
    #         j = sp[1]
    #         model.Add(sum(sum(z[(a, b)] for b in range(self.N) 
    #             if b != a and ((a == i and b == j) or (a == j or b == i))) 
    #             for a in range(self.N)) <= 0)

        # for sp in chunk:
        #     self.add_constraint(sp, chunk, self.N, model)
    # def calculate_sum(self, z, a, b, i, j):
    #     if b != a and ((a == i and b == j) or (a == j or b == i)):
    #         return z[(a, b)]
    #     return 0
    def process_chunk(self, chunk, z, model):
        for sp in chunk:
            i = sp[0]
            j = sp[1]
            model.Add(sum(sum(z[(a, b)] for b in range(self.N) 
                if b != a and ((a == i and b == j) or (a == j or b == i))) 
                for a in range(self.N)) <= 0)
        
    # model = cp_model.CpModel()
    # def add_constraints_batch(self, z, batch):
    #     for sp in batch:
    #         i = sp[0]
    #         j = sp[1]
    #         self.model.Add(sum(sum(z[(a, b)] for b in range(self.N) 
    #             if b != a and ((a == i and b == j) or (a == j or b == i))) 
    #             for a in range(self.N)) <= 0)
            


    def create_GORT_model(self, file_name:str):
        """Google OR Tools model"""

        
        # model = mp.Manager().Value(cp_model.CpModel())
        model = cp_model.CpModel()
        # Decision variables
        z = {}
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    z[(i, j)] = model.NewBoolVar(name='z' + str(i)+ ',' + str(j))
        
        
        # Constraints
        
        print()
        # chunk_size = len(self.same_photos) // 4
        # chunks = [self.same_photos[i:i+chunk_size] for i in range(0, len(self.same_photos), chunk_size)]
        # futures = []
        # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        #     for chunk in chunks:
        #         future = executor.submit(self.add_constraints, chunk, z, model)
                 
        #         futures.append(future)

        # concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

        # for chunk in chunks:
        #     self.add_constraints(chunk, z, model)


        # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        #     # Submit each iteration of the loop as a separate task to the executor
        #     tasks = [executor.submit(self.add_constraint, sp, z, self.N, model) for sp in self.same_photos]

        #     # Wait for all tasks to complete before continuing
        #     concurrent.futures.wait(tasks)

        # with ThreadPoolExecutor(max_workers=4) as executor:
        #     sum_list = []
        #     for a in range(self.N):
        #         for b in range(self.N):
        #             if a != b:
        #                 sum_list.append(executor.submit(self.calculate_sum, z, a, b, i, j))
            
        #     model.Add(sum(sum_list) <= 0)

        #     executor.wait()



            # initialize the model and variables here
    
    # split the same_photos list into chunks
        # num_chunks = 4
        # chunk_size = len(self.same_photos) // num_chunks
        # chunks = [self.same_photos[i:i+chunk_size] for i in range(0, len(self.same_photos), chunk_size)]
        # num_chunks = len(chunks)
        
        # process each chunk in parallel using threads
        # with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
        #     futures = []
        #     for chunk in chunks:
        #         future = executor.submit(self.process_chunk, chunk, z, model)
        #         futures.append(future)
            
        #     # wait for all threads to finish before continuing
        #     for future in concurrent.futures.as_completed(futures):
        #         result = future.result()

        
        # if __name__ == '__main__':
        #     with Pool(processes=4) as pool:
        #         chunk_size = len(self.same_photos) // 4
        #         chunks = [self.same_photos[i:i+chunk_size] for i in range(0, len(self.same_photos), chunk_size)]
        #         # pool.map(add_constraints_proccess, chunks, z, model, self.N)
        #         results = pool.starmap(add_constraints, [(z, chunk, self.N) for chunk in chunks])

        #         pool.close()
        #         pool.join()

        # for result in results:
        #     model.Add(result[0])
        # if __name__ == '__main__':
        #     with mp.Manager() as manager:
        #         model = manager. # create a shared model object

        #         # create a shared list to hold the futures
        #         futures = manager.list()

        #         # create the pool of processes
        #         with mp.Pool(processes=4) as pool:
        #             chunk_size = len(self.same_photos) // 4
        #             chunks = [self.same_photos[i:i+chunk_size] for i in range(0, len(self.same_photos), chunk_size)]
                    
        #             # submit tasks to the pool
        #             for chunk in chunks:
        #                 future = pool.apply_async(add_constraints_proccess, args=(chunk, self.N, model))
        #                 futures.append(future)

        #             # wait for the tasks to complete
        #             for future in futures:
        #                 future.get()


        # concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

        
        # if __name__ == '__main__':
        #     # manager = mp.Manager()
        #     # model_dict = manager.dict()
        #     # model_dict["model"] = model
        #     # lock = manager.Lock()
            
        #     with mp.Pool(processes=4) as pool:
        #         chunk_size = len(self.same_photos) // 4
        #         chunks = [self.same_photos[i:i+chunk_size] for i in range(0, len(self.same_photos), chunk_size)]
        #         results = pool.starmap(add_constraints, [(z, chunk, self.N) for chunk in chunks])
                
                
        #         pool.close()
        #         pool.join()

        #     for result in results:
        #         model.Add(result)

        
        # with mp.Pool(processes=4) as pool:
        #     # Submit tasks to the pool
        #     for chunk in chunks:
        #         pool.apply_async(self.add_constraints, args=(z, chunk, lock))

        #     # Wait for all tasks to complete
        #     pool.close()
        #     pool.join()


        # with mp.Pool(processes=4) as pool:
        #     results = [pool.apply_async(self.add_constraints, args=(chunk, vars)) for chunk in chunks]
        #     # Wait for subprocesses to finish and add their constraints to the main model
        #     # constraints = cp_model.cp_model_pb2.ConstraintProto()
        #     for result in results:
        #         # Parse the serialized constraints and add them to the main model
        #         model.Add(result.get())




        # if __name__ == '__main__':
        #     with Pool(processes=4) as pool:
        #         chunk_size = len(self.same_photos) // 4
        #         chunks = [self.same_photos[i:i+chunk_size] for i in range(0, len(self.same_photos), chunk_size)]
        #         results = []
        #         for chunk in chunks:
        #             results.append(pool.apply_async(add_constraints, args=(z, chunk, self.N)))
                

        #         pool.close()
        #         pool.join()
        #     for constraint in results:
        #         model.Add(constraint)


        # model_constraints = mp.Manager().list()
        # chunk_size = len(self.same_photos) // 100
        # chunks = [self.same_photos[i:i+chunk_size] for i in range(0, len(self.same_photos), chunk_size)]
        # results = []
        # if __name__ == '__main__':
        #     with Pool(processes=4) as pool:
                
        #         # results.append(pool.starmap(add_constraints, [(z, chunk, self.N) for chunk in chunks]))
        #         # nested_results = pool.starmap(add_constraints, [(z, chunk, self.N) for chunk in chunks])
        #         # results = list(itertools.chain.from_iterable(nested_results))
        #         results = pool.imap(add_constraints, [(z, chunk, self.N) for chunk in chunks])
        #         # constraints = pool.map(lambda sp: sum(sum(z[(a, b)] for b in range(N) 
        #         #     if b != a and ((a == sp[0] and b == sp[1]) or (a == sp[1] and b == sp[0])) for a in range(N)) <= 0),
        #         #     self.same_photos)



        #         pool.close()
        #         pool.join()

        # constraints = list(itertools.chain.from_iterable(results))

        # constraints = list(itertools.chain.from_iterable(results))
        # for c in constraints:
        #     model.Add(c)
        # num_processes = 4
        # # chunks = chunks(list(itertools.combinations(self.same_photos, 2)), num_processes)
        # chunk_size = len(self.same_photos) // num_processes
        # chunks = [self.same_photos[i:i+chunk_size] for i in range(0, len(self.same_photos), chunk_size)]

        # pool = mp.Pool(num_processes)
        # results = pool.starmap(self.add_constraints_batch, [(z, chunk) for chunk in chunks])
        # pool.close()
        # pool.join()
        
        
        # chunks = [self.same_photos[i:i + num_processes] for i in range(0, len(self.same_photos), num_processes)]
        # with Pool(processes= num_processes) as pool:
        #     constraints_list = pool.starmap(add_constraints, [(z, chunk, self.N) for chunk in chunks])
        #     pool.close()
        #     pool.join()
        
        # constraints = [c for sublist in constraints_list for c in sublist]
        # for c in constraints_list:
        #     for constraint in c:
        #         model.Add(constraint <= 0)
            
        
        
        # for p in processes:
        #     p.join()
        
        # model_constraints = []
        # while not results_queue.empty():
        #     constraints = results_queue.get()
        #     model_constraints.extend(constraints)

        # chunk_size = len(self.same_photos) // 100
        # chunks = [self.same_photos[i:i+chunk_size] for i in range(0, len(self.same_photos), chunk_size)]
        # results = []
        # if __name__ == '__main__':
        #     with Pool(processes=4) as pool:
        #         results = pool.imap(add_constraints, [(z, chunk, self.N) for chunk in chunks])
        #         pool.close()
        #         pool.join()

        # constraints = list(results)
        # for c in constraints:
        #     constraint_expr = json.loads(c['constraint'])
        #     operator = c['operator']
        #     value = int(c['value'])
        #     if operator == '<=':
        #         model.Add(constraint_expr <= value)



        chunk_size = len(self.same_photos) // 4
        chunks = [self.same_photos[i:i+chunk_size] for i in range(0, len(self.same_photos), chunk_size)]
        num_chunks = len(chunks)
        if __name__ == '__main__':
            counter = 0
            q = mp.Queue(maxsize=4)
            processes = []
            for i,chunk in enumerate(chunks):
                p = mp.Process(target=process_chunk, args=(z, q, chunk, self.N, counter))
                counter += 1
                processes.append(p)
                p.start()

            results = []
            for i in range(len(chunks)):
                result = q.get()
                results.append(result)

            for p in processes:
                p.join()

            print(results)



        # if __name__ == '__main__':
        #     with Pool(processes=4) as pool:
        #         q = mp.Queue()
        #         for i, chunk in enumerate(chunks):
        #             pool.apply_async(add_constraints, args=[z, chunk, self.N, i, q])
                
                

        #         while not q.full():
        #             pass

        #         pool.close()
        #         pool.join()


        # if __name__ == "__main__":
        #     # Set up the pool of workers and shared data
        #     num_processes = 4
        #     pool = Pool(processes=num_processes)
        #     manager = mp.Manager()
        #     results = manager.list()
            
        #     # Process each chunk asynchronously
        #     for i in range(10):
        #         same_photos_chunk = self.same_photos[i]
        #         file_suffix = f"{i:03d}"
        #         N = len(same_photos_chunk)
        #         args = (same_photos_chunk, N, file_suffix)
        #         async_result = pool.apply_async(process_chunk, args)
        #         results.append(async_result)
            
        #     # Wait for all processes to complete
        #     for async_result in results:
        #         result = async_result.get()
        #         add_constraints(result)
            
        #     # Clean up
        #     pool.close()
        #     pool.join()


        # Read in constraints from all files and add to model
        constraints = []
        for i in range(len(chunks)):
            with open(f"./constraint_chunks/constraints_{i}.txt", "r") as f:
                constraints += f.read().strip().split("\n")
        for c in constraints:
            model.Add(eval(c))




        # for c in constraints:
        #     model.Add(c)
        print()
        for sp in self.same_photos:
            i = sp[0]
            j = sp[1]
            model.Add(sum(sum(z[(a, b)] for b in range(self.N) 
                if b != a and ((a == i and b == j) or (a == j or b == i))) 
                for a in range(self.N)) <= 0)
            
        print()
        
        # for i in range(self.N ):
        #     model.Add(sum(z[(i, j)] for j in range(self.N) if j != i) <= 1)
        
        # for j in range(self.N):
        #     model.Add(sum(z[(i, j)] for i in range(self.N) if j != i) <= 1)
        
        # for i in range(self.N):
        #     for j in range(self.N):
        #         if i!=j:
        #             model.Add(z[(i,j)]+z[(j,i)] <= 1)


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


    def solveFromProto(self, file):
        with open(file, 'rb') as f:
            proto_data = f.read()

        # pdata = proto_data

        # with open('model.proto', 'r') as f:
        #     model_proto = cp_model_pb2.CpModelProto()
        #     text_format.Parse(f.read(), model_proto)

        
        model = cp_model.CpModel()
        model_proto = model.Proto()
        model_proto.ParseFromString(proto_data)
        
        
        # model.CopyFrom(model_proto)
        # x = text_format.Parse(proto_data, model)
        # model.CopyFrom(x)
        
        z = {}
        # for i in range(self.N):
        #     for j in range(self.N):
        #         if i != j:
        #             # z[(i, j)] = model.NewBoolVar(name='z' + str(i)+ ',' + str(j))
        #             model.GetBoolVarFromProtoIndex(i)

        z={}
        # test = model_proto.variables
        for i in range(len(model_proto.variables)):
            z[model_proto.variables[i].name] = model.GetBoolVarFromProtoIndex(i)


        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 1800

        status = solver.Solve(model)
        
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

def help_function():
    """Define help function"""
    print("The solver should be callde usign the command 'python photo_slideshow_solver.py instance_name.txt'")
    print("Example: python photo_slideshow_solver.py c_memorable_moments_50.txt")

if __name__=="__main__":
    """ Create an instance of the model and solve the problem"""
    # Recursion limit for the solver
    sys.setrecursionlimit(100000)

    arguments=sys.argv
    if len(arguments)!=2:
        help_function()
        exit()
    else:
        file_name = arguments[1]
        # toSave = "Hello World!"
        # with open('Results for 100 photos.txt', 'w') as f: 
        #     f.write(toSave)

        ps=PhotoSlideShow(file_name)
        
        # Gurobi model
        # model=ps.create_model()
        # objective_value=ps.solve_problem(model)
        # print("Objective value:: "+str(objective_value))
        

        # Googl OR Tools
        model=ps.create_GORT_model(os.path.splitext(file_name)[0])
        # model = ps.solveFromProto("./Model_100_Photos.proto")




# model = cp_model.CpModel()
