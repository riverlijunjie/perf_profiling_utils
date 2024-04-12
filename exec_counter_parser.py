
file_name = 'dino_exe_time.txt'

def validate_id(id, max_id):
    if id >=0 and id < max_id:
        return True
    else:
        return False

result = {}
primitive_cout = {}

with open(file_name) as f:
    for line in f:
        if 'realTime' in line and 'execType' in line:
            data = line.split()
            layer_type_id = data.index('layerType:')
            exec_type_id = data.index('execType:')
            real_time_id = data.index('realTime')
            cpu_time_id = data.index('cpuTime')
            #print(data)
            
            size = len(data)
            if validate_id(layer_type_id,size-1) and validate_id(exec_type_id,size-1) and validate_id(exec_type_id,size-2) and validate_id(cpu_time_id,size-2):
                key = (data[layer_type_id+1], data[exec_type_id+1])
                query_key = result.get(key)
                if query_key == None:
                    result[key] = float(data[real_time_id+2])
                    primitive_cout[key] = 1
                else:
                    result[key] += float(data[real_time_id+2])
                    primitive_cout[key] += 1
            #print(layer_type_id,exec_type_id)
            #print(data)
            #break

sorted_result = sorted(result.items(), key = lambda x:x[1], reverse=True)
#print(sorted_result)
total = 0.0
for k,v in sorted_result:
    total += v
for k,v in sorted_result:
    val = round(v,4)
    percent = round(v/total*100)
    res = ": "+str(val)+", "+str(percent)+str("%") + ", count = " + str(primitive_cout[k])
    #print(k,": ",v, " - ", v/total*100, "%")
    print(k,res,end="")
    print()
    
print("total = ", total, "ms", " fps =", 1000/total)
