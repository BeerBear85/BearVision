import json

required_parameter_fields = {'name', 'value', 'unit'}
allowed_parameter_fields = {'id','comment'}.union(required_parameter_fields)
allowed_units = {'s', 'frame', 'pixel','-','fps'}

#todo: 
#check for unique IDs
#add get by id


class ParameterHandler:
	def __init__(self, arg_filename, arg_subpart):
	
		f = open(arg_filename, 'r')
		self.full_parameters = json.load(f)
		f.close()
		self.param_list = self.full_parameters[arg_subpart]
		self.parameter_validation()
	
	def get_value_by_name(self, arg_input_string):
		for param in self.param_list:
			#print("param[name]:" + param["name"])
			if (param["name"] == arg_input_string):
				return param["value"]
		raise Exception("Error: " + arg_input_string + " was not found in the parameters!")

	def parameter_validation(self):
		for param in self.param_list:
			self.validate_parameter_fields(param)
			self.validate_unit(param)
					
	def validate_parameter_fields(self, arg_param):
		for key in arg_param.keys():
			if key not in allowed_parameter_fields:
				raise Exception("Field: \"" + key + "\" in " + arg_param["name"] + " is not a valid parameter field!")
		for required_key in required_parameter_fields:
			if required_key not in arg_param.keys():
				raise Exception("The required field: \"" + required_key + "\" in the parameter " + arg_param["name"] + " has not been found!")
	
	def validate_unit(self, arg_param):
		if arg_param['unit'] not in allowed_units:
			raise Exception("Unit: \"" + arg_param['unit'] + "\" in the parameter" + arg_param["name"] + " is not a valid unit!")