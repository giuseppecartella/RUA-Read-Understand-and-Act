import numpy as np

def _isDiagonal(paths):
        THRESHOLD = 2

        A = paths[0]
        B = paths[1]
        C = paths[2]

        angle_AB = int(np.degrees(np.arctan( (B[1] - A[1]) / (B[0] - A[0] + 0.00001) )))
        angle_BC = int(np.degrees(np.arctan( (C[1] - B[1]) / (C[0] - B[0] + 0.00001) )))

        return abs(angle_AB - angle_BC) < THRESHOLD

def _avoid_diagonal(paths):
        shrink_paths = []
        i = 0
        was_diagonal = False    
        first_point = None      #First diagonal's point
        
        while i < len(paths) - 2:
            is_diagonal = _isDiagonal(paths[i:i+3])
            
            if is_diagonal == True:
                if first_point is None:
                    first_point = paths[i]
                last_point = paths[i+2]
                was_diagonal = True
            elif (not is_diagonal) and (was_diagonal == True):
                shrink_paths.append(first_point)
                shrink_paths.append(last_point)
                was_diagonal = False
                first_point = None
                #i = i + 1
            else:
                shrink_paths.append(paths[i])
            i = i + 1
        
        # ---- To HANDLE LAST TRIPLET ----#
        if was_diagonal:
            shrink_paths.append(paths[-1])
        else:
            if paths[0] not in shrink_paths:
                shrink_paths.insert(0, paths[0])
            
            if (first_point is not None) and (first_point not in shrink_paths):
                shrink_paths.append(first_point)
            else:
                shrink_paths.append(paths[-2])
            
            if paths[-1] not in shrink_paths:
                shrink_paths.append(paths[-1])
        
        shrink_paths = np.array(shrink_paths)
        shrink_paths = np.unique(shrink_paths, axis=0)

        return shrink_paths




paths = [(0,50),(10,50),(15,55),(20,60), (20,80), (20,100), (30,80)] #alto, diagonale in alto verso destra, poi tutto a destra
#paths = [(0,50),(10,50),(20,60), (30,70), (30,80),(30,90), (30,100)]
#paths = [(0,50),(10,50),(20,50),(50,50), (60,50), (100,100), (40,80)] #dritto poi tutto a destra
path = _avoid_diagonal(paths)
print(path)



