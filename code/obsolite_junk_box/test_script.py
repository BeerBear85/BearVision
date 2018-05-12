import json
import sys
sys.path.append('modules')
import ParameterHandler


para = ParameterHandler.ParameterHandler('parameters.json','Preprocessor')



#print("Parameters: " + str(parameters))
#preprocessor_parameters = parameters["Preprocessor"]
#print(preprocessor_parameters[0])
#print( json.dumps(parameters, indent=3) )
#print(preprocessor_parameters[0].keys())


my_value = para.get_value_by_name("start_frame")
print("Value: " + str(my_value))




#JSON	Python
#object	dict
#array	list
#string	str
#number (int)	int
#number (real)	float
#true	True
#false	False
#null	None