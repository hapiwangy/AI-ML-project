½

á
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	

ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018¿Û
ª
)Adam/auto_encoder/decoder/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/auto_encoder/decoder/conv2d_6/bias/v
£
=Adam/auto_encoder/decoder/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOp)Adam/auto_encoder/decoder/conv2d_6/bias/v*
_output_shapes
:*
dtype0
º
+Adam/auto_encoder/decoder/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/auto_encoder/decoder/conv2d_6/kernel/v
³
?Adam/auto_encoder/decoder/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/auto_encoder/decoder/conv2d_6/kernel/v*&
_output_shapes
: *
dtype0
ª
)Adam/auto_encoder/decoder/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)Adam/auto_encoder/decoder/conv2d_5/bias/v
£
=Adam/auto_encoder/decoder/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOp)Adam/auto_encoder/decoder/conv2d_5/bias/v*
_output_shapes
: *
dtype0
º
+Adam/auto_encoder/decoder/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *<
shared_name-+Adam/auto_encoder/decoder/conv2d_5/kernel/v
³
?Adam/auto_encoder/decoder/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/auto_encoder/decoder/conv2d_5/kernel/v*&
_output_shapes
:  *
dtype0
ª
)Adam/auto_encoder/decoder/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)Adam/auto_encoder/decoder/conv2d_4/bias/v
£
=Adam/auto_encoder/decoder/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOp)Adam/auto_encoder/decoder/conv2d_4/bias/v*
_output_shapes
: *
dtype0
º
+Adam/auto_encoder/decoder/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/auto_encoder/decoder/conv2d_4/kernel/v
³
?Adam/auto_encoder/decoder/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/auto_encoder/decoder/conv2d_4/kernel/v*&
_output_shapes
: *
dtype0
ª
)Adam/auto_encoder/decoder/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/auto_encoder/decoder/conv2d_3/bias/v
£
=Adam/auto_encoder/decoder/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOp)Adam/auto_encoder/decoder/conv2d_3/bias/v*
_output_shapes
:*
dtype0
º
+Adam/auto_encoder/decoder/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/auto_encoder/decoder/conv2d_3/kernel/v
³
?Adam/auto_encoder/decoder/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/auto_encoder/decoder/conv2d_3/kernel/v*&
_output_shapes
:*
dtype0
ª
)Adam/auto_encoder/encoder/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/auto_encoder/encoder/conv2d_2/bias/v
£
=Adam/auto_encoder/encoder/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOp)Adam/auto_encoder/encoder/conv2d_2/bias/v*
_output_shapes
:*
dtype0
º
+Adam/auto_encoder/encoder/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/auto_encoder/encoder/conv2d_2/kernel/v
³
?Adam/auto_encoder/encoder/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/auto_encoder/encoder/conv2d_2/kernel/v*&
_output_shapes
: *
dtype0
ª
)Adam/auto_encoder/encoder/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)Adam/auto_encoder/encoder/conv2d_1/bias/v
£
=Adam/auto_encoder/encoder/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOp)Adam/auto_encoder/encoder/conv2d_1/bias/v*
_output_shapes
: *
dtype0
º
+Adam/auto_encoder/encoder/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *<
shared_name-+Adam/auto_encoder/encoder/conv2d_1/kernel/v
³
?Adam/auto_encoder/encoder/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/auto_encoder/encoder/conv2d_1/kernel/v*&
_output_shapes
:  *
dtype0
¦
'Adam/auto_encoder/encoder/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adam/auto_encoder/encoder/conv2d/bias/v

;Adam/auto_encoder/encoder/conv2d/bias/v/Read/ReadVariableOpReadVariableOp'Adam/auto_encoder/encoder/conv2d/bias/v*
_output_shapes
: *
dtype0
¶
)Adam/auto_encoder/encoder/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)Adam/auto_encoder/encoder/conv2d/kernel/v
¯
=Adam/auto_encoder/encoder/conv2d/kernel/v/Read/ReadVariableOpReadVariableOp)Adam/auto_encoder/encoder/conv2d/kernel/v*&
_output_shapes
: *
dtype0
ª
)Adam/auto_encoder/decoder/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/auto_encoder/decoder/conv2d_6/bias/m
£
=Adam/auto_encoder/decoder/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOp)Adam/auto_encoder/decoder/conv2d_6/bias/m*
_output_shapes
:*
dtype0
º
+Adam/auto_encoder/decoder/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/auto_encoder/decoder/conv2d_6/kernel/m
³
?Adam/auto_encoder/decoder/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/auto_encoder/decoder/conv2d_6/kernel/m*&
_output_shapes
: *
dtype0
ª
)Adam/auto_encoder/decoder/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)Adam/auto_encoder/decoder/conv2d_5/bias/m
£
=Adam/auto_encoder/decoder/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOp)Adam/auto_encoder/decoder/conv2d_5/bias/m*
_output_shapes
: *
dtype0
º
+Adam/auto_encoder/decoder/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *<
shared_name-+Adam/auto_encoder/decoder/conv2d_5/kernel/m
³
?Adam/auto_encoder/decoder/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/auto_encoder/decoder/conv2d_5/kernel/m*&
_output_shapes
:  *
dtype0
ª
)Adam/auto_encoder/decoder/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)Adam/auto_encoder/decoder/conv2d_4/bias/m
£
=Adam/auto_encoder/decoder/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOp)Adam/auto_encoder/decoder/conv2d_4/bias/m*
_output_shapes
: *
dtype0
º
+Adam/auto_encoder/decoder/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/auto_encoder/decoder/conv2d_4/kernel/m
³
?Adam/auto_encoder/decoder/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/auto_encoder/decoder/conv2d_4/kernel/m*&
_output_shapes
: *
dtype0
ª
)Adam/auto_encoder/decoder/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/auto_encoder/decoder/conv2d_3/bias/m
£
=Adam/auto_encoder/decoder/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOp)Adam/auto_encoder/decoder/conv2d_3/bias/m*
_output_shapes
:*
dtype0
º
+Adam/auto_encoder/decoder/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+Adam/auto_encoder/decoder/conv2d_3/kernel/m
³
?Adam/auto_encoder/decoder/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/auto_encoder/decoder/conv2d_3/kernel/m*&
_output_shapes
:*
dtype0
ª
)Adam/auto_encoder/encoder/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/auto_encoder/encoder/conv2d_2/bias/m
£
=Adam/auto_encoder/encoder/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOp)Adam/auto_encoder/encoder/conv2d_2/bias/m*
_output_shapes
:*
dtype0
º
+Adam/auto_encoder/encoder/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/auto_encoder/encoder/conv2d_2/kernel/m
³
?Adam/auto_encoder/encoder/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/auto_encoder/encoder/conv2d_2/kernel/m*&
_output_shapes
: *
dtype0
ª
)Adam/auto_encoder/encoder/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)Adam/auto_encoder/encoder/conv2d_1/bias/m
£
=Adam/auto_encoder/encoder/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOp)Adam/auto_encoder/encoder/conv2d_1/bias/m*
_output_shapes
: *
dtype0
º
+Adam/auto_encoder/encoder/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *<
shared_name-+Adam/auto_encoder/encoder/conv2d_1/kernel/m
³
?Adam/auto_encoder/encoder/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/auto_encoder/encoder/conv2d_1/kernel/m*&
_output_shapes
:  *
dtype0
¦
'Adam/auto_encoder/encoder/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adam/auto_encoder/encoder/conv2d/bias/m

;Adam/auto_encoder/encoder/conv2d/bias/m/Read/ReadVariableOpReadVariableOp'Adam/auto_encoder/encoder/conv2d/bias/m*
_output_shapes
: *
dtype0
¶
)Adam/auto_encoder/encoder/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)Adam/auto_encoder/encoder/conv2d/kernel/m
¯
=Adam/auto_encoder/encoder/conv2d/kernel/m/Read/ReadVariableOpReadVariableOp)Adam/auto_encoder/encoder/conv2d/kernel/m*&
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

"auto_encoder/decoder/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"auto_encoder/decoder/conv2d_6/bias

6auto_encoder/decoder/conv2d_6/bias/Read/ReadVariableOpReadVariableOp"auto_encoder/decoder/conv2d_6/bias*
_output_shapes
:*
dtype0
¬
$auto_encoder/decoder/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$auto_encoder/decoder/conv2d_6/kernel
¥
8auto_encoder/decoder/conv2d_6/kernel/Read/ReadVariableOpReadVariableOp$auto_encoder/decoder/conv2d_6/kernel*&
_output_shapes
: *
dtype0

"auto_encoder/decoder/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"auto_encoder/decoder/conv2d_5/bias

6auto_encoder/decoder/conv2d_5/bias/Read/ReadVariableOpReadVariableOp"auto_encoder/decoder/conv2d_5/bias*
_output_shapes
: *
dtype0
¬
$auto_encoder/decoder/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *5
shared_name&$auto_encoder/decoder/conv2d_5/kernel
¥
8auto_encoder/decoder/conv2d_5/kernel/Read/ReadVariableOpReadVariableOp$auto_encoder/decoder/conv2d_5/kernel*&
_output_shapes
:  *
dtype0

"auto_encoder/decoder/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"auto_encoder/decoder/conv2d_4/bias

6auto_encoder/decoder/conv2d_4/bias/Read/ReadVariableOpReadVariableOp"auto_encoder/decoder/conv2d_4/bias*
_output_shapes
: *
dtype0
¬
$auto_encoder/decoder/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$auto_encoder/decoder/conv2d_4/kernel
¥
8auto_encoder/decoder/conv2d_4/kernel/Read/ReadVariableOpReadVariableOp$auto_encoder/decoder/conv2d_4/kernel*&
_output_shapes
: *
dtype0

"auto_encoder/decoder/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"auto_encoder/decoder/conv2d_3/bias

6auto_encoder/decoder/conv2d_3/bias/Read/ReadVariableOpReadVariableOp"auto_encoder/decoder/conv2d_3/bias*
_output_shapes
:*
dtype0
¬
$auto_encoder/decoder/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$auto_encoder/decoder/conv2d_3/kernel
¥
8auto_encoder/decoder/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp$auto_encoder/decoder/conv2d_3/kernel*&
_output_shapes
:*
dtype0

"auto_encoder/encoder/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"auto_encoder/encoder/conv2d_2/bias

6auto_encoder/encoder/conv2d_2/bias/Read/ReadVariableOpReadVariableOp"auto_encoder/encoder/conv2d_2/bias*
_output_shapes
:*
dtype0
¬
$auto_encoder/encoder/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$auto_encoder/encoder/conv2d_2/kernel
¥
8auto_encoder/encoder/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp$auto_encoder/encoder/conv2d_2/kernel*&
_output_shapes
: *
dtype0

"auto_encoder/encoder/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"auto_encoder/encoder/conv2d_1/bias

6auto_encoder/encoder/conv2d_1/bias/Read/ReadVariableOpReadVariableOp"auto_encoder/encoder/conv2d_1/bias*
_output_shapes
: *
dtype0
¬
$auto_encoder/encoder/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *5
shared_name&$auto_encoder/encoder/conv2d_1/kernel
¥
8auto_encoder/encoder/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp$auto_encoder/encoder/conv2d_1/kernel*&
_output_shapes
:  *
dtype0

 auto_encoder/encoder/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" auto_encoder/encoder/conv2d/bias

4auto_encoder/encoder/conv2d/bias/Read/ReadVariableOpReadVariableOp auto_encoder/encoder/conv2d/bias*
_output_shapes
: *
dtype0
¨
"auto_encoder/encoder/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"auto_encoder/encoder/conv2d/kernel
¡
6auto_encoder/encoder/conv2d/kernel/Read/ReadVariableOpReadVariableOp"auto_encoder/encoder/conv2d/kernel*&
_output_shapes
: *
dtype0

NoOpNoOp
Ùj
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*j
valuejBj Bj
ð
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
loss
	encoder

decoder
	optimizer

signatures*
j
0
1
2
3
4
5
6
7
8
9
10
11
12
13*
j
0
1
2
3
4
5
6
7
8
9
10
11
12
13*
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

 trace_0
!trace_1* 

"trace_0
#trace_1* 
* 
* 
»
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
	*conv1
	+conv2
	,conv3
-pool*
Ê
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
	4conv1
	5conv2
	6conv3
	7conv4
8upsample*
Ü
9iter

:beta_1

;beta_2
	<decay
=learning_ratemÀmÁmÂmÃmÄmÅmÆmÇmÈmÉmÊmËmÌmÍvÎvÏvÐvÑvÒvÓvÔvÕvÖv×vØvÙvÚvÛ*

>serving_default* 
b\
VARIABLE_VALUE"auto_encoder/encoder/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE auto_encoder/encoder/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$auto_encoder/encoder/conv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"auto_encoder/encoder/conv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$auto_encoder/encoder/conv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"auto_encoder/encoder/conv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$auto_encoder/decoder/conv2d_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"auto_encoder/decoder/conv2d_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE$auto_encoder/decoder/conv2d_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"auto_encoder/decoder/conv2d_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$auto_encoder/decoder/conv2d_5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"auto_encoder/decoder/conv2d_5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE$auto_encoder/decoder/conv2d_6/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE"auto_encoder/decoder/conv2d_6/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
* 

	0

1*

?0*
* 
* 
* 
* 
* 
* 
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 

@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

Etrace_0* 

Ftrace_0* 
È
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

kernel
bias
 M_jit_compiled_convolution_op*
È
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

kernel
bias
 T_jit_compiled_convolution_op*
È
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

kernel
bias
 [_jit_compiled_convolution_op*

\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses* 
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 

bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

gtrace_0* 

htrace_0* 
È
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

kernel
bias
 o_jit_compiled_convolution_op*
È
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses

kernel
bias
 v_jit_compiled_convolution_op*
È
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

kernel
bias
 }_jit_compiled_convolution_op*
Í
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias
!_jit_compiled_convolution_op*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
	variables
	keras_api

total

count*
* 
 
*0
+1
,2
-3*
* 
* 
* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*
* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*
* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses* 

£trace_0* 

¤trace_0* 
* 
'
40
51
62
73
84*
* 
* 
* 
* 
* 

0
1*

0
1*
* 

¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*
* 
* 
* 

0
1*

0
1*
* 

ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses*
* 
* 
* 

0
1*

0
1*
* 

¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*
* 
* 
* 

0
1*

0
1*
* 

´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

¾trace_0* 

¿trace_0* 

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

VARIABLE_VALUE)Adam/auto_encoder/encoder/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE'Adam/auto_encoder/encoder/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/auto_encoder/encoder/conv2d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/auto_encoder/encoder/conv2d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/auto_encoder/encoder/conv2d_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/auto_encoder/encoder/conv2d_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/auto_encoder/decoder/conv2d_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/auto_encoder/decoder/conv2d_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/auto_encoder/decoder/conv2d_4/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/auto_encoder/decoder/conv2d_4/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/auto_encoder/decoder/conv2d_5/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/auto_encoder/decoder/conv2d_5/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/auto_encoder/decoder/conv2d_6/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/auto_encoder/decoder/conv2d_6/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/auto_encoder/encoder/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE'Adam/auto_encoder/encoder/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/auto_encoder/encoder/conv2d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/auto_encoder/encoder/conv2d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/auto_encoder/encoder/conv2d_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/auto_encoder/encoder/conv2d_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/auto_encoder/decoder/conv2d_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/auto_encoder/decoder/conv2d_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/auto_encoder/decoder/conv2d_4/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/auto_encoder/decoder/conv2d_4/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/auto_encoder/decoder/conv2d_5/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/auto_encoder/decoder/conv2d_5/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE+Adam/auto_encoder/decoder/conv2d_6/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)Adam/auto_encoder/decoder/conv2d_6/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_1Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
Ú
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1"auto_encoder/encoder/conv2d/kernel auto_encoder/encoder/conv2d/bias$auto_encoder/encoder/conv2d_1/kernel"auto_encoder/encoder/conv2d_1/bias$auto_encoder/encoder/conv2d_2/kernel"auto_encoder/encoder/conv2d_2/bias$auto_encoder/decoder/conv2d_3/kernel"auto_encoder/decoder/conv2d_3/bias$auto_encoder/decoder/conv2d_4/kernel"auto_encoder/decoder/conv2d_4/bias$auto_encoder/decoder/conv2d_5/kernel"auto_encoder/decoder/conv2d_5/bias$auto_encoder/decoder/conv2d_6/kernel"auto_encoder/decoder/conv2d_6/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_84922
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¿
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename6auto_encoder/encoder/conv2d/kernel/Read/ReadVariableOp4auto_encoder/encoder/conv2d/bias/Read/ReadVariableOp8auto_encoder/encoder/conv2d_1/kernel/Read/ReadVariableOp6auto_encoder/encoder/conv2d_1/bias/Read/ReadVariableOp8auto_encoder/encoder/conv2d_2/kernel/Read/ReadVariableOp6auto_encoder/encoder/conv2d_2/bias/Read/ReadVariableOp8auto_encoder/decoder/conv2d_3/kernel/Read/ReadVariableOp6auto_encoder/decoder/conv2d_3/bias/Read/ReadVariableOp8auto_encoder/decoder/conv2d_4/kernel/Read/ReadVariableOp6auto_encoder/decoder/conv2d_4/bias/Read/ReadVariableOp8auto_encoder/decoder/conv2d_5/kernel/Read/ReadVariableOp6auto_encoder/decoder/conv2d_5/bias/Read/ReadVariableOp8auto_encoder/decoder/conv2d_6/kernel/Read/ReadVariableOp6auto_encoder/decoder/conv2d_6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp=Adam/auto_encoder/encoder/conv2d/kernel/m/Read/ReadVariableOp;Adam/auto_encoder/encoder/conv2d/bias/m/Read/ReadVariableOp?Adam/auto_encoder/encoder/conv2d_1/kernel/m/Read/ReadVariableOp=Adam/auto_encoder/encoder/conv2d_1/bias/m/Read/ReadVariableOp?Adam/auto_encoder/encoder/conv2d_2/kernel/m/Read/ReadVariableOp=Adam/auto_encoder/encoder/conv2d_2/bias/m/Read/ReadVariableOp?Adam/auto_encoder/decoder/conv2d_3/kernel/m/Read/ReadVariableOp=Adam/auto_encoder/decoder/conv2d_3/bias/m/Read/ReadVariableOp?Adam/auto_encoder/decoder/conv2d_4/kernel/m/Read/ReadVariableOp=Adam/auto_encoder/decoder/conv2d_4/bias/m/Read/ReadVariableOp?Adam/auto_encoder/decoder/conv2d_5/kernel/m/Read/ReadVariableOp=Adam/auto_encoder/decoder/conv2d_5/bias/m/Read/ReadVariableOp?Adam/auto_encoder/decoder/conv2d_6/kernel/m/Read/ReadVariableOp=Adam/auto_encoder/decoder/conv2d_6/bias/m/Read/ReadVariableOp=Adam/auto_encoder/encoder/conv2d/kernel/v/Read/ReadVariableOp;Adam/auto_encoder/encoder/conv2d/bias/v/Read/ReadVariableOp?Adam/auto_encoder/encoder/conv2d_1/kernel/v/Read/ReadVariableOp=Adam/auto_encoder/encoder/conv2d_1/bias/v/Read/ReadVariableOp?Adam/auto_encoder/encoder/conv2d_2/kernel/v/Read/ReadVariableOp=Adam/auto_encoder/encoder/conv2d_2/bias/v/Read/ReadVariableOp?Adam/auto_encoder/decoder/conv2d_3/kernel/v/Read/ReadVariableOp=Adam/auto_encoder/decoder/conv2d_3/bias/v/Read/ReadVariableOp?Adam/auto_encoder/decoder/conv2d_4/kernel/v/Read/ReadVariableOp=Adam/auto_encoder/decoder/conv2d_4/bias/v/Read/ReadVariableOp?Adam/auto_encoder/decoder/conv2d_5/kernel/v/Read/ReadVariableOp=Adam/auto_encoder/decoder/conv2d_5/bias/v/Read/ReadVariableOp?Adam/auto_encoder/decoder/conv2d_6/kernel/v/Read/ReadVariableOp=Adam/auto_encoder/decoder/conv2d_6/bias/v/Read/ReadVariableOpConst*>
Tin7
523	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_85330
æ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename"auto_encoder/encoder/conv2d/kernel auto_encoder/encoder/conv2d/bias$auto_encoder/encoder/conv2d_1/kernel"auto_encoder/encoder/conv2d_1/bias$auto_encoder/encoder/conv2d_2/kernel"auto_encoder/encoder/conv2d_2/bias$auto_encoder/decoder/conv2d_3/kernel"auto_encoder/decoder/conv2d_3/bias$auto_encoder/decoder/conv2d_4/kernel"auto_encoder/decoder/conv2d_4/bias$auto_encoder/decoder/conv2d_5/kernel"auto_encoder/decoder/conv2d_5/bias$auto_encoder/decoder/conv2d_6/kernel"auto_encoder/decoder/conv2d_6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount)Adam/auto_encoder/encoder/conv2d/kernel/m'Adam/auto_encoder/encoder/conv2d/bias/m+Adam/auto_encoder/encoder/conv2d_1/kernel/m)Adam/auto_encoder/encoder/conv2d_1/bias/m+Adam/auto_encoder/encoder/conv2d_2/kernel/m)Adam/auto_encoder/encoder/conv2d_2/bias/m+Adam/auto_encoder/decoder/conv2d_3/kernel/m)Adam/auto_encoder/decoder/conv2d_3/bias/m+Adam/auto_encoder/decoder/conv2d_4/kernel/m)Adam/auto_encoder/decoder/conv2d_4/bias/m+Adam/auto_encoder/decoder/conv2d_5/kernel/m)Adam/auto_encoder/decoder/conv2d_5/bias/m+Adam/auto_encoder/decoder/conv2d_6/kernel/m)Adam/auto_encoder/decoder/conv2d_6/bias/m)Adam/auto_encoder/encoder/conv2d/kernel/v'Adam/auto_encoder/encoder/conv2d/bias/v+Adam/auto_encoder/encoder/conv2d_1/kernel/v)Adam/auto_encoder/encoder/conv2d_1/bias/v+Adam/auto_encoder/encoder/conv2d_2/kernel/v)Adam/auto_encoder/encoder/conv2d_2/bias/v+Adam/auto_encoder/decoder/conv2d_3/kernel/v)Adam/auto_encoder/decoder/conv2d_3/bias/v+Adam/auto_encoder/decoder/conv2d_4/kernel/v)Adam/auto_encoder/decoder/conv2d_4/bias/v+Adam/auto_encoder/decoder/conv2d_5/kernel/v)Adam/auto_encoder/decoder/conv2d_5/bias/v+Adam/auto_encoder/decoder/conv2d_6/kernel/v)Adam/auto_encoder/decoder/conv2d_6/bias/v*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_85487ÇÂ
Ø

G__inference_auto_encoder_layer_call_and_return_conditional_losses_84881
input_1'
encoder_84850: 
encoder_84852: '
encoder_84854:  
encoder_84856: '
encoder_84858: 
encoder_84860:'
decoder_84863:
decoder_84865:'
decoder_84867: 
decoder_84869: '
decoder_84871:  
decoder_84873: '
decoder_84875: 
decoder_84877:
identity¢decoder/StatefulPartitionedCall¢encoder/StatefulPartitionedCall¹
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_84850encoder_84852encoder_84854encoder_84856encoder_84858encoder_84860*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_84666ü
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_84863decoder_84865decoder_84867decoder_84869decoder_84871decoder_84873decoder_84875decoder_84877*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_84724
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_85143

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦"

B__inference_encoder_layer_call_and_return_conditional_losses_84666
input_features?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:  6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource:
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¯
conv2d/Conv2DConv2Dinput_features$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Ã
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ «
max_pooling2d/MaxPool_1MaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Å
conv2d_2/Conv2DConv2D max_pooling2d/MaxPool_1:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
max_pooling2d/MaxPool_2MaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
w
IdentityIdentity max_pooling2d/MaxPool_2:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinput_features

d
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_85160

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹	
 
'__inference_encoder_layer_call_fn_85040
input_features!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: 
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_featuresunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_84666w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinput_features


×
'__inference_decoder_layer_call_fn_85089
encoded!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallencodedunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_84724w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	encoded
q
¶
 __inference__wrapped_model_84600
input_1T
:auto_encoder_encoder_conv2d_conv2d_readvariableop_resource: I
;auto_encoder_encoder_conv2d_biasadd_readvariableop_resource: V
<auto_encoder_encoder_conv2d_1_conv2d_readvariableop_resource:  K
=auto_encoder_encoder_conv2d_1_biasadd_readvariableop_resource: V
<auto_encoder_encoder_conv2d_2_conv2d_readvariableop_resource: K
=auto_encoder_encoder_conv2d_2_biasadd_readvariableop_resource:V
<auto_encoder_decoder_conv2d_3_conv2d_readvariableop_resource:K
=auto_encoder_decoder_conv2d_3_biasadd_readvariableop_resource:V
<auto_encoder_decoder_conv2d_4_conv2d_readvariableop_resource: K
=auto_encoder_decoder_conv2d_4_biasadd_readvariableop_resource: V
<auto_encoder_decoder_conv2d_5_conv2d_readvariableop_resource:  K
=auto_encoder_decoder_conv2d_5_biasadd_readvariableop_resource: V
<auto_encoder_decoder_conv2d_6_conv2d_readvariableop_resource: K
=auto_encoder_decoder_conv2d_6_biasadd_readvariableop_resource:
identity¢4auto_encoder/decoder/conv2d_3/BiasAdd/ReadVariableOp¢3auto_encoder/decoder/conv2d_3/Conv2D/ReadVariableOp¢4auto_encoder/decoder/conv2d_4/BiasAdd/ReadVariableOp¢3auto_encoder/decoder/conv2d_4/Conv2D/ReadVariableOp¢4auto_encoder/decoder/conv2d_5/BiasAdd/ReadVariableOp¢3auto_encoder/decoder/conv2d_5/Conv2D/ReadVariableOp¢4auto_encoder/decoder/conv2d_6/BiasAdd/ReadVariableOp¢3auto_encoder/decoder/conv2d_6/Conv2D/ReadVariableOp¢2auto_encoder/encoder/conv2d/BiasAdd/ReadVariableOp¢1auto_encoder/encoder/conv2d/Conv2D/ReadVariableOp¢4auto_encoder/encoder/conv2d_1/BiasAdd/ReadVariableOp¢3auto_encoder/encoder/conv2d_1/Conv2D/ReadVariableOp¢4auto_encoder/encoder/conv2d_2/BiasAdd/ReadVariableOp¢3auto_encoder/encoder/conv2d_2/Conv2D/ReadVariableOp´
1auto_encoder/encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp:auto_encoder_encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ò
"auto_encoder/encoder/conv2d/Conv2DConv2Dinput_19auto_encoder/encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
ª
2auto_encoder/encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp;auto_encoder_encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ñ
#auto_encoder/encoder/conv2d/BiasAddBiasAdd+auto_encoder/encoder/conv2d/Conv2D:output:0:auto_encoder/encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 auto_encoder/encoder/conv2d/ReluRelu,auto_encoder/encoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Ñ
*auto_encoder/encoder/max_pooling2d/MaxPoolMaxPool.auto_encoder/encoder/conv2d/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides
¸
3auto_encoder/encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp<auto_encoder_encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
$auto_encoder/encoder/conv2d_1/Conv2DConv2D3auto_encoder/encoder/max_pooling2d/MaxPool:output:0;auto_encoder/encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
®
4auto_encoder/encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp=auto_encoder_encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0×
%auto_encoder/encoder/conv2d_1/BiasAddBiasAdd-auto_encoder/encoder/conv2d_1/Conv2D:output:0<auto_encoder/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"auto_encoder/encoder/conv2d_1/ReluRelu.auto_encoder/encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Õ
,auto_encoder/encoder/max_pooling2d/MaxPool_1MaxPool0auto_encoder/encoder/conv2d_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides
¸
3auto_encoder/encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp<auto_encoder_encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
$auto_encoder/encoder/conv2d_2/Conv2DConv2D5auto_encoder/encoder/max_pooling2d/MaxPool_1:output:0;auto_encoder/encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
®
4auto_encoder/encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp=auto_encoder_encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0×
%auto_encoder/encoder/conv2d_2/BiasAddBiasAdd-auto_encoder/encoder/conv2d_2/Conv2D:output:0<auto_encoder/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"auto_encoder/encoder/conv2d_2/ReluRelu.auto_encoder/encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
,auto_encoder/encoder/max_pooling2d/MaxPool_2MaxPool0auto_encoder/encoder/conv2d_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
¸
3auto_encoder/decoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp<auto_encoder_decoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
$auto_encoder/decoder/conv2d_3/Conv2DConv2D5auto_encoder/encoder/max_pooling2d/MaxPool_2:output:0;auto_encoder/decoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
®
4auto_encoder/decoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp=auto_encoder_decoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0×
%auto_encoder/decoder/conv2d_3/BiasAddBiasAdd-auto_encoder/decoder/conv2d_3/Conv2D:output:0<auto_encoder/decoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"auto_encoder/decoder/conv2d_3/ReluRelu.auto_encoder/decoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
(auto_encoder/decoder/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      {
*auto_encoder/decoder/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      º
&auto_encoder/decoder/up_sampling2d/mulMul1auto_encoder/decoder/up_sampling2d/Const:output:03auto_encoder/decoder/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:
?auto_encoder/decoder/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor0auto_encoder/decoder/conv2d_3/Relu:activations:0*auto_encoder/decoder/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(¸
3auto_encoder/decoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp<auto_encoder_decoder_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
$auto_encoder/decoder/conv2d_4/Conv2DConv2DPauto_encoder/decoder/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0;auto_encoder/decoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
®
4auto_encoder/decoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp=auto_encoder_decoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0×
%auto_encoder/decoder/conv2d_4/BiasAddBiasAdd-auto_encoder/decoder/conv2d_4/Conv2D:output:0<auto_encoder/decoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"auto_encoder/decoder/conv2d_4/ReluRelu.auto_encoder/decoder/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
*auto_encoder/decoder/up_sampling2d/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      {
*auto_encoder/decoder/up_sampling2d/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      ¾
(auto_encoder/decoder/up_sampling2d/mul_1Mul3auto_encoder/decoder/up_sampling2d/Const_2:output:03auto_encoder/decoder/up_sampling2d/Const_3:output:0*
T0*
_output_shapes
:
Aauto_encoder/decoder/up_sampling2d/resize_1/ResizeNearestNeighborResizeNearestNeighbor0auto_encoder/decoder/conv2d_4/Relu:activations:0,auto_encoder/decoder/up_sampling2d/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(¸
3auto_encoder/decoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOp<auto_encoder_decoder_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0¢
$auto_encoder/decoder/conv2d_5/Conv2DConv2DRauto_encoder/decoder/up_sampling2d/resize_1/ResizeNearestNeighbor:resized_images:0;auto_encoder/decoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
®
4auto_encoder/decoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp=auto_encoder_decoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0×
%auto_encoder/decoder/conv2d_5/BiasAddBiasAdd-auto_encoder/decoder/conv2d_5/Conv2D:output:0<auto_encoder/decoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"auto_encoder/decoder/conv2d_5/ReluRelu.auto_encoder/decoder/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ {
*auto_encoder/decoder/up_sampling2d/Const_4Const*
_output_shapes
:*
dtype0*
valueB"      {
*auto_encoder/decoder/up_sampling2d/Const_5Const*
_output_shapes
:*
dtype0*
valueB"      ¾
(auto_encoder/decoder/up_sampling2d/mul_2Mul3auto_encoder/decoder/up_sampling2d/Const_4:output:03auto_encoder/decoder/up_sampling2d/Const_5:output:0*
T0*
_output_shapes
:
Aauto_encoder/decoder/up_sampling2d/resize_2/ResizeNearestNeighborResizeNearestNeighbor0auto_encoder/decoder/conv2d_5/Relu:activations:0,auto_encoder/decoder/up_sampling2d/mul_2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(¸
3auto_encoder/decoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp<auto_encoder_decoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¡
$auto_encoder/decoder/conv2d_6/Conv2DConv2DRauto_encoder/decoder/up_sampling2d/resize_2/ResizeNearestNeighbor:resized_images:0;auto_encoder/decoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
®
4auto_encoder/decoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp=auto_encoder_decoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0×
%auto_encoder/decoder/conv2d_6/BiasAddBiasAdd-auto_encoder/decoder/conv2d_6/Conv2D:output:0<auto_encoder/decoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%auto_encoder/decoder/conv2d_6/SigmoidSigmoid.auto_encoder/decoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity)auto_encoder/decoder/conv2d_6/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
NoOpNoOp5^auto_encoder/decoder/conv2d_3/BiasAdd/ReadVariableOp4^auto_encoder/decoder/conv2d_3/Conv2D/ReadVariableOp5^auto_encoder/decoder/conv2d_4/BiasAdd/ReadVariableOp4^auto_encoder/decoder/conv2d_4/Conv2D/ReadVariableOp5^auto_encoder/decoder/conv2d_5/BiasAdd/ReadVariableOp4^auto_encoder/decoder/conv2d_5/Conv2D/ReadVariableOp5^auto_encoder/decoder/conv2d_6/BiasAdd/ReadVariableOp4^auto_encoder/decoder/conv2d_6/Conv2D/ReadVariableOp3^auto_encoder/encoder/conv2d/BiasAdd/ReadVariableOp2^auto_encoder/encoder/conv2d/Conv2D/ReadVariableOp5^auto_encoder/encoder/conv2d_1/BiasAdd/ReadVariableOp4^auto_encoder/encoder/conv2d_1/Conv2D/ReadVariableOp5^auto_encoder/encoder/conv2d_2/BiasAdd/ReadVariableOp4^auto_encoder/encoder/conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2l
4auto_encoder/decoder/conv2d_3/BiasAdd/ReadVariableOp4auto_encoder/decoder/conv2d_3/BiasAdd/ReadVariableOp2j
3auto_encoder/decoder/conv2d_3/Conv2D/ReadVariableOp3auto_encoder/decoder/conv2d_3/Conv2D/ReadVariableOp2l
4auto_encoder/decoder/conv2d_4/BiasAdd/ReadVariableOp4auto_encoder/decoder/conv2d_4/BiasAdd/ReadVariableOp2j
3auto_encoder/decoder/conv2d_4/Conv2D/ReadVariableOp3auto_encoder/decoder/conv2d_4/Conv2D/ReadVariableOp2l
4auto_encoder/decoder/conv2d_5/BiasAdd/ReadVariableOp4auto_encoder/decoder/conv2d_5/BiasAdd/ReadVariableOp2j
3auto_encoder/decoder/conv2d_5/Conv2D/ReadVariableOp3auto_encoder/decoder/conv2d_5/Conv2D/ReadVariableOp2l
4auto_encoder/decoder/conv2d_6/BiasAdd/ReadVariableOp4auto_encoder/decoder/conv2d_6/BiasAdd/ReadVariableOp2j
3auto_encoder/decoder/conv2d_6/Conv2D/ReadVariableOp3auto_encoder/decoder/conv2d_6/Conv2D/ReadVariableOp2h
2auto_encoder/encoder/conv2d/BiasAdd/ReadVariableOp2auto_encoder/encoder/conv2d/BiasAdd/ReadVariableOp2f
1auto_encoder/encoder/conv2d/Conv2D/ReadVariableOp1auto_encoder/encoder/conv2d/Conv2D/ReadVariableOp2l
4auto_encoder/encoder/conv2d_1/BiasAdd/ReadVariableOp4auto_encoder/encoder/conv2d_1/BiasAdd/ReadVariableOp2j
3auto_encoder/encoder/conv2d_1/Conv2D/ReadVariableOp3auto_encoder/encoder/conv2d_1/Conv2D/ReadVariableOp2l
4auto_encoder/encoder/conv2d_2/BiasAdd/ReadVariableOp4auto_encoder/encoder/conv2d_2/BiasAdd/ReadVariableOp2j
3auto_encoder/encoder/conv2d_2/Conv2D/ReadVariableOp3auto_encoder/encoder/conv2d_2/Conv2D/ReadVariableOp:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_84609

inputs
identity¡
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÄÍ
å%
!__inference__traced_restore_85487
file_prefixM
3assignvariableop_auto_encoder_encoder_conv2d_kernel: A
3assignvariableop_1_auto_encoder_encoder_conv2d_bias: Q
7assignvariableop_2_auto_encoder_encoder_conv2d_1_kernel:  C
5assignvariableop_3_auto_encoder_encoder_conv2d_1_bias: Q
7assignvariableop_4_auto_encoder_encoder_conv2d_2_kernel: C
5assignvariableop_5_auto_encoder_encoder_conv2d_2_bias:Q
7assignvariableop_6_auto_encoder_decoder_conv2d_3_kernel:C
5assignvariableop_7_auto_encoder_decoder_conv2d_3_bias:Q
7assignvariableop_8_auto_encoder_decoder_conv2d_4_kernel: C
5assignvariableop_9_auto_encoder_decoder_conv2d_4_bias: R
8assignvariableop_10_auto_encoder_decoder_conv2d_5_kernel:  D
6assignvariableop_11_auto_encoder_decoder_conv2d_5_bias: R
8assignvariableop_12_auto_encoder_decoder_conv2d_6_kernel: D
6assignvariableop_13_auto_encoder_decoder_conv2d_6_bias:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: #
assignvariableop_19_total: #
assignvariableop_20_count: W
=assignvariableop_21_adam_auto_encoder_encoder_conv2d_kernel_m: I
;assignvariableop_22_adam_auto_encoder_encoder_conv2d_bias_m: Y
?assignvariableop_23_adam_auto_encoder_encoder_conv2d_1_kernel_m:  K
=assignvariableop_24_adam_auto_encoder_encoder_conv2d_1_bias_m: Y
?assignvariableop_25_adam_auto_encoder_encoder_conv2d_2_kernel_m: K
=assignvariableop_26_adam_auto_encoder_encoder_conv2d_2_bias_m:Y
?assignvariableop_27_adam_auto_encoder_decoder_conv2d_3_kernel_m:K
=assignvariableop_28_adam_auto_encoder_decoder_conv2d_3_bias_m:Y
?assignvariableop_29_adam_auto_encoder_decoder_conv2d_4_kernel_m: K
=assignvariableop_30_adam_auto_encoder_decoder_conv2d_4_bias_m: Y
?assignvariableop_31_adam_auto_encoder_decoder_conv2d_5_kernel_m:  K
=assignvariableop_32_adam_auto_encoder_decoder_conv2d_5_bias_m: Y
?assignvariableop_33_adam_auto_encoder_decoder_conv2d_6_kernel_m: K
=assignvariableop_34_adam_auto_encoder_decoder_conv2d_6_bias_m:W
=assignvariableop_35_adam_auto_encoder_encoder_conv2d_kernel_v: I
;assignvariableop_36_adam_auto_encoder_encoder_conv2d_bias_v: Y
?assignvariableop_37_adam_auto_encoder_encoder_conv2d_1_kernel_v:  K
=assignvariableop_38_adam_auto_encoder_encoder_conv2d_1_bias_v: Y
?assignvariableop_39_adam_auto_encoder_encoder_conv2d_2_kernel_v: K
=assignvariableop_40_adam_auto_encoder_encoder_conv2d_2_bias_v:Y
?assignvariableop_41_adam_auto_encoder_decoder_conv2d_3_kernel_v:K
=assignvariableop_42_adam_auto_encoder_decoder_conv2d_3_bias_v:Y
?assignvariableop_43_adam_auto_encoder_decoder_conv2d_4_kernel_v: K
=assignvariableop_44_adam_auto_encoder_decoder_conv2d_4_bias_v: Y
?assignvariableop_45_adam_auto_encoder_decoder_conv2d_5_kernel_v:  K
=assignvariableop_46_adam_auto_encoder_decoder_conv2d_5_bias_v: Y
?assignvariableop_47_adam_auto_encoder_decoder_conv2d_6_kernel_v: K
=assignvariableop_48_adam_auto_encoder_decoder_conv2d_6_bias_v:
identity_50¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*°
value¦B£2B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÔ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Þ
_output_shapesË
È::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp3assignvariableop_auto_encoder_encoder_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_1AssignVariableOp3assignvariableop_1_auto_encoder_encoder_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_2AssignVariableOp7assignvariableop_2_auto_encoder_encoder_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_3AssignVariableOp5assignvariableop_3_auto_encoder_encoder_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_4AssignVariableOp7assignvariableop_4_auto_encoder_encoder_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_5AssignVariableOp5assignvariableop_5_auto_encoder_encoder_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_6AssignVariableOp7assignvariableop_6_auto_encoder_decoder_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_7AssignVariableOp5assignvariableop_7_auto_encoder_decoder_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_8AssignVariableOp7assignvariableop_8_auto_encoder_decoder_conv2d_4_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_9AssignVariableOp5assignvariableop_9_auto_encoder_decoder_conv2d_4_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_10AssignVariableOp8assignvariableop_10_auto_encoder_decoder_conv2d_5_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_11AssignVariableOp6assignvariableop_11_auto_encoder_decoder_conv2d_5_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_12AssignVariableOp8assignvariableop_12_auto_encoder_decoder_conv2d_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_13AssignVariableOp6assignvariableop_13_auto_encoder_decoder_conv2d_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_21AssignVariableOp=assignvariableop_21_adam_auto_encoder_encoder_conv2d_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_22AssignVariableOp;assignvariableop_22_adam_auto_encoder_encoder_conv2d_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_23AssignVariableOp?assignvariableop_23_adam_auto_encoder_encoder_conv2d_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_24AssignVariableOp=assignvariableop_24_adam_auto_encoder_encoder_conv2d_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_25AssignVariableOp?assignvariableop_25_adam_auto_encoder_encoder_conv2d_2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_26AssignVariableOp=assignvariableop_26_adam_auto_encoder_encoder_conv2d_2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_27AssignVariableOp?assignvariableop_27_adam_auto_encoder_decoder_conv2d_3_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_28AssignVariableOp=assignvariableop_28_adam_auto_encoder_decoder_conv2d_3_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_29AssignVariableOp?assignvariableop_29_adam_auto_encoder_decoder_conv2d_4_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_30AssignVariableOp=assignvariableop_30_adam_auto_encoder_decoder_conv2d_4_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_31AssignVariableOp?assignvariableop_31_adam_auto_encoder_decoder_conv2d_5_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_32AssignVariableOp=assignvariableop_32_adam_auto_encoder_decoder_conv2d_5_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_33AssignVariableOp?assignvariableop_33_adam_auto_encoder_decoder_conv2d_6_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_34AssignVariableOp=assignvariableop_34_adam_auto_encoder_decoder_conv2d_6_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_35AssignVariableOp=assignvariableop_35_adam_auto_encoder_encoder_conv2d_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_36AssignVariableOp;assignvariableop_36_adam_auto_encoder_encoder_conv2d_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_37AssignVariableOp?assignvariableop_37_adam_auto_encoder_encoder_conv2d_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_38AssignVariableOp=assignvariableop_38_adam_auto_encoder_encoder_conv2d_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_39AssignVariableOp?assignvariableop_39_adam_auto_encoder_encoder_conv2d_2_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_40AssignVariableOp=assignvariableop_40_adam_auto_encoder_encoder_conv2d_2_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_41AssignVariableOp?assignvariableop_41_adam_auto_encoder_decoder_conv2d_3_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_42AssignVariableOp=assignvariableop_42_adam_auto_encoder_decoder_conv2d_3_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_43AssignVariableOp?assignvariableop_43_adam_auto_encoder_decoder_conv2d_4_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_44AssignVariableOp=assignvariableop_44_adam_auto_encoder_decoder_conv2d_4_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_45AssignVariableOp?assignvariableop_45_adam_auto_encoder_decoder_conv2d_5_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_46AssignVariableOp=assignvariableop_46_adam_auto_encoder_decoder_conv2d_5_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_47AssignVariableOp?assignvariableop_47_adam_auto_encoder_decoder_conv2d_6_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_48AssignVariableOp=assignvariableop_48_adam_auto_encoder_decoder_conv2d_6_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_50IdentityIdentity_49:output:0^NoOp_1*
T0*
_output_shapes
: ò
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_50Identity_50:output:0*w
_input_shapesf
d: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ám
÷
__inference__traced_save_85330
file_prefixA
=savev2_auto_encoder_encoder_conv2d_kernel_read_readvariableop?
;savev2_auto_encoder_encoder_conv2d_bias_read_readvariableopC
?savev2_auto_encoder_encoder_conv2d_1_kernel_read_readvariableopA
=savev2_auto_encoder_encoder_conv2d_1_bias_read_readvariableopC
?savev2_auto_encoder_encoder_conv2d_2_kernel_read_readvariableopA
=savev2_auto_encoder_encoder_conv2d_2_bias_read_readvariableopC
?savev2_auto_encoder_decoder_conv2d_3_kernel_read_readvariableopA
=savev2_auto_encoder_decoder_conv2d_3_bias_read_readvariableopC
?savev2_auto_encoder_decoder_conv2d_4_kernel_read_readvariableopA
=savev2_auto_encoder_decoder_conv2d_4_bias_read_readvariableopC
?savev2_auto_encoder_decoder_conv2d_5_kernel_read_readvariableopA
=savev2_auto_encoder_decoder_conv2d_5_bias_read_readvariableopC
?savev2_auto_encoder_decoder_conv2d_6_kernel_read_readvariableopA
=savev2_auto_encoder_decoder_conv2d_6_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopH
Dsavev2_adam_auto_encoder_encoder_conv2d_kernel_m_read_readvariableopF
Bsavev2_adam_auto_encoder_encoder_conv2d_bias_m_read_readvariableopJ
Fsavev2_adam_auto_encoder_encoder_conv2d_1_kernel_m_read_readvariableopH
Dsavev2_adam_auto_encoder_encoder_conv2d_1_bias_m_read_readvariableopJ
Fsavev2_adam_auto_encoder_encoder_conv2d_2_kernel_m_read_readvariableopH
Dsavev2_adam_auto_encoder_encoder_conv2d_2_bias_m_read_readvariableopJ
Fsavev2_adam_auto_encoder_decoder_conv2d_3_kernel_m_read_readvariableopH
Dsavev2_adam_auto_encoder_decoder_conv2d_3_bias_m_read_readvariableopJ
Fsavev2_adam_auto_encoder_decoder_conv2d_4_kernel_m_read_readvariableopH
Dsavev2_adam_auto_encoder_decoder_conv2d_4_bias_m_read_readvariableopJ
Fsavev2_adam_auto_encoder_decoder_conv2d_5_kernel_m_read_readvariableopH
Dsavev2_adam_auto_encoder_decoder_conv2d_5_bias_m_read_readvariableopJ
Fsavev2_adam_auto_encoder_decoder_conv2d_6_kernel_m_read_readvariableopH
Dsavev2_adam_auto_encoder_decoder_conv2d_6_bias_m_read_readvariableopH
Dsavev2_adam_auto_encoder_encoder_conv2d_kernel_v_read_readvariableopF
Bsavev2_adam_auto_encoder_encoder_conv2d_bias_v_read_readvariableopJ
Fsavev2_adam_auto_encoder_encoder_conv2d_1_kernel_v_read_readvariableopH
Dsavev2_adam_auto_encoder_encoder_conv2d_1_bias_v_read_readvariableopJ
Fsavev2_adam_auto_encoder_encoder_conv2d_2_kernel_v_read_readvariableopH
Dsavev2_adam_auto_encoder_encoder_conv2d_2_bias_v_read_readvariableopJ
Fsavev2_adam_auto_encoder_decoder_conv2d_3_kernel_v_read_readvariableopH
Dsavev2_adam_auto_encoder_decoder_conv2d_3_bias_v_read_readvariableopJ
Fsavev2_adam_auto_encoder_decoder_conv2d_4_kernel_v_read_readvariableopH
Dsavev2_adam_auto_encoder_decoder_conv2d_4_bias_v_read_readvariableopJ
Fsavev2_adam_auto_encoder_decoder_conv2d_5_kernel_v_read_readvariableopH
Dsavev2_adam_auto_encoder_decoder_conv2d_5_bias_v_read_readvariableopJ
Fsavev2_adam_auto_encoder_decoder_conv2d_6_kernel_v_read_readvariableopH
Dsavev2_adam_auto_encoder_decoder_conv2d_6_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*°
value¦B£2B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÑ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ©
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0=savev2_auto_encoder_encoder_conv2d_kernel_read_readvariableop;savev2_auto_encoder_encoder_conv2d_bias_read_readvariableop?savev2_auto_encoder_encoder_conv2d_1_kernel_read_readvariableop=savev2_auto_encoder_encoder_conv2d_1_bias_read_readvariableop?savev2_auto_encoder_encoder_conv2d_2_kernel_read_readvariableop=savev2_auto_encoder_encoder_conv2d_2_bias_read_readvariableop?savev2_auto_encoder_decoder_conv2d_3_kernel_read_readvariableop=savev2_auto_encoder_decoder_conv2d_3_bias_read_readvariableop?savev2_auto_encoder_decoder_conv2d_4_kernel_read_readvariableop=savev2_auto_encoder_decoder_conv2d_4_bias_read_readvariableop?savev2_auto_encoder_decoder_conv2d_5_kernel_read_readvariableop=savev2_auto_encoder_decoder_conv2d_5_bias_read_readvariableop?savev2_auto_encoder_decoder_conv2d_6_kernel_read_readvariableop=savev2_auto_encoder_decoder_conv2d_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopDsavev2_adam_auto_encoder_encoder_conv2d_kernel_m_read_readvariableopBsavev2_adam_auto_encoder_encoder_conv2d_bias_m_read_readvariableopFsavev2_adam_auto_encoder_encoder_conv2d_1_kernel_m_read_readvariableopDsavev2_adam_auto_encoder_encoder_conv2d_1_bias_m_read_readvariableopFsavev2_adam_auto_encoder_encoder_conv2d_2_kernel_m_read_readvariableopDsavev2_adam_auto_encoder_encoder_conv2d_2_bias_m_read_readvariableopFsavev2_adam_auto_encoder_decoder_conv2d_3_kernel_m_read_readvariableopDsavev2_adam_auto_encoder_decoder_conv2d_3_bias_m_read_readvariableopFsavev2_adam_auto_encoder_decoder_conv2d_4_kernel_m_read_readvariableopDsavev2_adam_auto_encoder_decoder_conv2d_4_bias_m_read_readvariableopFsavev2_adam_auto_encoder_decoder_conv2d_5_kernel_m_read_readvariableopDsavev2_adam_auto_encoder_decoder_conv2d_5_bias_m_read_readvariableopFsavev2_adam_auto_encoder_decoder_conv2d_6_kernel_m_read_readvariableopDsavev2_adam_auto_encoder_decoder_conv2d_6_bias_m_read_readvariableopDsavev2_adam_auto_encoder_encoder_conv2d_kernel_v_read_readvariableopBsavev2_adam_auto_encoder_encoder_conv2d_bias_v_read_readvariableopFsavev2_adam_auto_encoder_encoder_conv2d_1_kernel_v_read_readvariableopDsavev2_adam_auto_encoder_encoder_conv2d_1_bias_v_read_readvariableopFsavev2_adam_auto_encoder_encoder_conv2d_2_kernel_v_read_readvariableopDsavev2_adam_auto_encoder_encoder_conv2d_2_bias_v_read_readvariableopFsavev2_adam_auto_encoder_decoder_conv2d_3_kernel_v_read_readvariableopDsavev2_adam_auto_encoder_decoder_conv2d_3_bias_v_read_readvariableopFsavev2_adam_auto_encoder_decoder_conv2d_4_kernel_v_read_readvariableopDsavev2_adam_auto_encoder_decoder_conv2d_4_bias_v_read_readvariableopFsavev2_adam_auto_encoder_decoder_conv2d_5_kernel_v_read_readvariableopDsavev2_adam_auto_encoder_decoder_conv2d_5_bias_v_read_readvariableopFsavev2_adam_auto_encoder_decoder_conv2d_6_kernel_v_read_readvariableopDsavev2_adam_auto_encoder_decoder_conv2d_6_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: : : :  : : :::: : :  : : :: : : : : : : : : :  : : :::: : :  : : :: : :  : : :::: : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
: : 


_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :, (
&
_output_shapes
:  : !

_output_shapes
: :,"(
&
_output_shapes
: : #

_output_shapes
::,$(
&
_output_shapes
: : %

_output_shapes
: :,&(
&
_output_shapes
:  : '

_output_shapes
: :,((
&
_output_shapes
: : )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
: : -

_output_shapes
: :,.(
&
_output_shapes
:  : /

_output_shapes
: :,0(
&
_output_shapes
: : 1

_output_shapes
::2

_output_shapes
: 
µ
I
-__inference_max_pooling2d_layer_call_fn_85138

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_84609
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
I
-__inference_up_sampling2d_layer_call_fn_85148

inputs
identityÙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_84628
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø4
Ú
B__inference_decoder_layer_call_and_return_conditional_losses_84724
encodedA
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:A
'conv2d_4_conv2d_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource: A
'conv2d_5_conv2d_readvariableop_resource:  6
(conv2d_5_biasadd_readvariableop_resource: A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource:
identity¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢conv2d_4/BiasAdd/ReadVariableOp¢conv2d_4/Conv2D/ReadVariableOp¢conv2d_5/BiasAdd/ReadVariableOp¢conv2d_5/Conv2D/ReadVariableOp¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¬
conv2d_3/Conv2DConv2Dencoded&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      f
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      {
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:Ë
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0à
conv2d_4/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ f
up_sampling2d/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      f
up_sampling2d/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d/mul_1Mulup_sampling2d/Const_2:output:0up_sampling2d/Const_3:output:0*
T0*
_output_shapes
:Ï
,up_sampling2d/resize_1/ResizeNearestNeighborResizeNearestNeighborconv2d_4/Relu:activations:0up_sampling2d/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0ã
conv2d_5/Conv2DConv2D=up_sampling2d/resize_1/ResizeNearestNeighbor:resized_images:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ f
up_sampling2d/Const_4Const*
_output_shapes
:*
dtype0*
valueB"      f
up_sampling2d/Const_5Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d/mul_2Mulup_sampling2d/Const_4:output:0up_sampling2d/Const_5:output:0*
T0*
_output_shapes
:Ï
,up_sampling2d/resize_2/ResizeNearestNeighborResizeNearestNeighborconv2d_5/Relu:activations:0up_sampling2d/mul_2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0â
conv2d_6/Conv2DConv2D=up_sampling2d/resize_2/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
conv2d_6/SigmoidSigmoidconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
IdentityIdentityconv2d_6/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	encoded

d
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_84628

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:µ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
 
,__inference_auto_encoder_layer_call_fn_84955
input_features!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: 
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: $

unknown_11: 

unknown_12:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_featuresunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_auto_encoder_layer_call_and_return_conditional_losses_84743w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinput_features
æ^
ø
G__inference_auto_encoder_layer_call_and_return_conditional_losses_85023
input_featuresG
-encoder_conv2d_conv2d_readvariableop_resource: <
.encoder_conv2d_biasadd_readvariableop_resource: I
/encoder_conv2d_1_conv2d_readvariableop_resource:  >
0encoder_conv2d_1_biasadd_readvariableop_resource: I
/encoder_conv2d_2_conv2d_readvariableop_resource: >
0encoder_conv2d_2_biasadd_readvariableop_resource:I
/decoder_conv2d_3_conv2d_readvariableop_resource:>
0decoder_conv2d_3_biasadd_readvariableop_resource:I
/decoder_conv2d_4_conv2d_readvariableop_resource: >
0decoder_conv2d_4_biasadd_readvariableop_resource: I
/decoder_conv2d_5_conv2d_readvariableop_resource:  >
0decoder_conv2d_5_biasadd_readvariableop_resource: I
/decoder_conv2d_6_conv2d_readvariableop_resource: >
0decoder_conv2d_6_biasadd_readvariableop_resource:
identity¢'decoder/conv2d_3/BiasAdd/ReadVariableOp¢&decoder/conv2d_3/Conv2D/ReadVariableOp¢'decoder/conv2d_4/BiasAdd/ReadVariableOp¢&decoder/conv2d_4/Conv2D/ReadVariableOp¢'decoder/conv2d_5/BiasAdd/ReadVariableOp¢&decoder/conv2d_5/Conv2D/ReadVariableOp¢'decoder/conv2d_6/BiasAdd/ReadVariableOp¢&decoder/conv2d_6/Conv2D/ReadVariableOp¢%encoder/conv2d/BiasAdd/ReadVariableOp¢$encoder/conv2d/Conv2D/ReadVariableOp¢'encoder/conv2d_1/BiasAdd/ReadVariableOp¢&encoder/conv2d_1/Conv2D/ReadVariableOp¢'encoder/conv2d_2/BiasAdd/ReadVariableOp¢&encoder/conv2d_2/Conv2D/ReadVariableOp
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¿
encoder/conv2d/Conv2DConv2Dinput_features,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ª
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ v
encoder/conv2d/ReluReluencoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ·
encoder/max_pooling2d/MaxPoolMaxPool!encoder/conv2d/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides

&encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Û
encoder/conv2d_1/Conv2DConv2D&encoder/max_pooling2d/MaxPool:output:0.encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

'encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0°
encoder/conv2d_1/BiasAddBiasAdd encoder/conv2d_1/Conv2D:output:0/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
encoder/conv2d_1/ReluRelu!encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ »
encoder/max_pooling2d/MaxPool_1MaxPool#encoder/conv2d_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides

&encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Ý
encoder/conv2d_2/Conv2DConv2D(encoder/max_pooling2d/MaxPool_1:output:0.encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

'encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0°
encoder/conv2d_2/BiasAddBiasAdd encoder/conv2d_2/Conv2D:output:0/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
encoder/conv2d_2/ReluRelu!encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
encoder/max_pooling2d/MaxPool_2MaxPool#encoder/conv2d_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides

&decoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ý
decoder/conv2d_3/Conv2DConv2D(encoder/max_pooling2d/MaxPool_2:output:0.decoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

'decoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0°
decoder/conv2d_3/BiasAddBiasAdd decoder/conv2d_3/Conv2D:output:0/decoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
decoder/conv2d_3/ReluRelu!decoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
decoder/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      n
decoder/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      
decoder/up_sampling2d/mulMul$decoder/up_sampling2d/Const:output:0&decoder/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:ã
2decoder/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor#decoder/conv2d_3/Relu:activations:0decoder/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
&decoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ø
decoder/conv2d_4/Conv2DConv2DCdecoder/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0.decoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

'decoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0°
decoder/conv2d_4/BiasAddBiasAdd decoder/conv2d_4/Conv2D:output:0/decoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
decoder/conv2d_4/ReluRelu!decoder/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
decoder/up_sampling2d/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      n
decoder/up_sampling2d/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      
decoder/up_sampling2d/mul_1Mul&decoder/up_sampling2d/Const_2:output:0&decoder/up_sampling2d/Const_3:output:0*
T0*
_output_shapes
:ç
4decoder/up_sampling2d/resize_1/ResizeNearestNeighborResizeNearestNeighbor#decoder/conv2d_4/Relu:activations:0decoder/up_sampling2d/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(
&decoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0û
decoder/conv2d_5/Conv2DConv2DEdecoder/up_sampling2d/resize_1/ResizeNearestNeighbor:resized_images:0.decoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

'decoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0°
decoder/conv2d_5/BiasAddBiasAdd decoder/conv2d_5/Conv2D:output:0/decoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
decoder/conv2d_5/ReluRelu!decoder/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
decoder/up_sampling2d/Const_4Const*
_output_shapes
:*
dtype0*
valueB"      n
decoder/up_sampling2d/Const_5Const*
_output_shapes
:*
dtype0*
valueB"      
decoder/up_sampling2d/mul_2Mul&decoder/up_sampling2d/Const_4:output:0&decoder/up_sampling2d/Const_5:output:0*
T0*
_output_shapes
:ç
4decoder/up_sampling2d/resize_2/ResizeNearestNeighborResizeNearestNeighbor#decoder/conv2d_5/Relu:activations:0decoder/up_sampling2d/mul_2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(
&decoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ú
decoder/conv2d_6/Conv2DConv2DEdecoder/up_sampling2d/resize_2/ResizeNearestNeighbor:resized_images:0.decoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

'decoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0°
decoder/conv2d_6/BiasAddBiasAdd decoder/conv2d_6/Conv2D:output:0/decoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
decoder/conv2d_6/SigmoidSigmoid!decoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
IdentityIdentitydecoder/conv2d_6/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp(^decoder/conv2d_3/BiasAdd/ReadVariableOp'^decoder/conv2d_3/Conv2D/ReadVariableOp(^decoder/conv2d_4/BiasAdd/ReadVariableOp'^decoder/conv2d_4/Conv2D/ReadVariableOp(^decoder/conv2d_5/BiasAdd/ReadVariableOp'^decoder/conv2d_5/Conv2D/ReadVariableOp(^decoder/conv2d_6/BiasAdd/ReadVariableOp'^decoder/conv2d_6/Conv2D/ReadVariableOp&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2R
'decoder/conv2d_3/BiasAdd/ReadVariableOp'decoder/conv2d_3/BiasAdd/ReadVariableOp2P
&decoder/conv2d_3/Conv2D/ReadVariableOp&decoder/conv2d_3/Conv2D/ReadVariableOp2R
'decoder/conv2d_4/BiasAdd/ReadVariableOp'decoder/conv2d_4/BiasAdd/ReadVariableOp2P
&decoder/conv2d_4/Conv2D/ReadVariableOp&decoder/conv2d_4/Conv2D/ReadVariableOp2R
'decoder/conv2d_5/BiasAdd/ReadVariableOp'decoder/conv2d_5/BiasAdd/ReadVariableOp2P
&decoder/conv2d_5/Conv2D/ReadVariableOp&decoder/conv2d_5/Conv2D/ReadVariableOp2R
'decoder/conv2d_6/BiasAdd/ReadVariableOp'decoder/conv2d_6/BiasAdd/ReadVariableOp2P
&decoder/conv2d_6/Conv2D/ReadVariableOp&decoder/conv2d_6/Conv2D/ReadVariableOp2N
%encoder/conv2d/BiasAdd/ReadVariableOp%encoder/conv2d/BiasAdd/ReadVariableOp2L
$encoder/conv2d/Conv2D/ReadVariableOp$encoder/conv2d/Conv2D/ReadVariableOp2R
'encoder/conv2d_1/BiasAdd/ReadVariableOp'encoder/conv2d_1/BiasAdd/ReadVariableOp2P
&encoder/conv2d_1/Conv2D/ReadVariableOp&encoder/conv2d_1/Conv2D/ReadVariableOp2R
'encoder/conv2d_2/BiasAdd/ReadVariableOp'encoder/conv2d_2/BiasAdd/ReadVariableOp2P
&encoder/conv2d_2/Conv2D/ReadVariableOp&encoder/conv2d_2/Conv2D/ReadVariableOp:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinput_features
¤

,__inference_auto_encoder_layer_call_fn_84774
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: 
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: $

unknown_11: 

unknown_12:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_auto_encoder_layer_call_and_return_conditional_losses_84743w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ø4
Ú
B__inference_decoder_layer_call_and_return_conditional_losses_85133
encodedA
'conv2d_3_conv2d_readvariableop_resource:6
(conv2d_3_biasadd_readvariableop_resource:A
'conv2d_4_conv2d_readvariableop_resource: 6
(conv2d_4_biasadd_readvariableop_resource: A
'conv2d_5_conv2d_readvariableop_resource:  6
(conv2d_5_biasadd_readvariableop_resource: A
'conv2d_6_conv2d_readvariableop_resource: 6
(conv2d_6_biasadd_readvariableop_resource:
identity¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢conv2d_4/BiasAdd/ReadVariableOp¢conv2d_4/Conv2D/ReadVariableOp¢conv2d_5/BiasAdd/ReadVariableOp¢conv2d_5/Conv2D/ReadVariableOp¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¬
conv2d_3/Conv2DConv2Dencoded&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      f
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      {
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:Ë
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0à
conv2d_4/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ f
up_sampling2d/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      f
up_sampling2d/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d/mul_1Mulup_sampling2d/Const_2:output:0up_sampling2d/Const_3:output:0*
T0*
_output_shapes
:Ï
,up_sampling2d/resize_1/ResizeNearestNeighborResizeNearestNeighborconv2d_4/Relu:activations:0up_sampling2d/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0ã
conv2d_5/Conv2DConv2D=up_sampling2d/resize_1/ResizeNearestNeighbor:resized_images:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides

conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ f
up_sampling2d/Const_4Const*
_output_shapes
:*
dtype0*
valueB"      f
up_sampling2d/Const_5Const*
_output_shapes
:*
dtype0*
valueB"      
up_sampling2d/mul_2Mulup_sampling2d/Const_4:output:0up_sampling2d/Const_5:output:0*
T0*
_output_shapes
:Ï
,up_sampling2d/resize_2/ResizeNearestNeighborResizeNearestNeighborconv2d_5/Relu:activations:0up_sampling2d/mul_2:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0â
conv2d_6/Conv2DConv2D=up_sampling2d/resize_2/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
conv2d_6/SigmoidSigmoidconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
IdentityIdentityconv2d_6/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	encoded
¦"

B__inference_encoder_layer_call_and_return_conditional_losses_85068
input_features?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource:  6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource:
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¯
conv2d/Conv2DConv2Dinput_features$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ §
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Ã
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ «
max_pooling2d/MaxPool_1MaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Å
conv2d_2/Conv2DConv2D max_pooling2d/MaxPool_1:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
max_pooling2d/MaxPool_2MaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
w
IdentityIdentity max_pooling2d/MaxPool_2:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinput_features
í

G__inference_auto_encoder_layer_call_and_return_conditional_losses_84743
input_features'
encoder_84667: 
encoder_84669: '
encoder_84671:  
encoder_84673: '
encoder_84675: 
encoder_84677:'
decoder_84725:
decoder_84727:'
decoder_84729: 
decoder_84731: '
decoder_84733:  
decoder_84735: '
decoder_84737: 
decoder_84739:
identity¢decoder/StatefulPartitionedCall¢encoder/StatefulPartitionedCallÀ
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_featuresencoder_84667encoder_84669encoder_84671encoder_84673encoder_84675encoder_84677*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_84666ü
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_84725decoder_84727decoder_84729decoder_84731decoder_84733decoder_84735decoder_84737decoder_84739*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_84724
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:_ [
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinput_features
ô

#__inference_signature_wrapper_84922
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3: 
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: $

unknown_11: 

unknown_12:
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_84600w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*»
serving_default§
C
input_18
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿD
output_18
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ÆË

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
loss
	encoder

decoder
	optimizer

signatures"
_tf_keras_model

0
1
2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper

0
1
2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
À
 trace_0
!trace_12
,__inference_auto_encoder_layer_call_fn_84774
,__inference_auto_encoder_layer_call_fn_84955ª
¡²
FullArgSpec%
args
jself
jinput_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z trace_0z!trace_1
ö
"trace_0
#trace_12¿
G__inference_auto_encoder_layer_call_and_return_conditional_losses_85023
G__inference_auto_encoder_layer_call_and_return_conditional_losses_84881ª
¡²
FullArgSpec%
args
jself
jinput_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z"trace_0z#trace_1
ËBÈ
 __inference__wrapped_model_84600input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
Ð
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
	*conv1
	+conv2
	,conv3
-pool"
_tf_keras_layer
ß
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
	4conv1
	5conv2
	6conv3
	7conv4
8upsample"
_tf_keras_layer
ë
9iter

:beta_1

;beta_2
	<decay
=learning_ratemÀmÁmÂmÃmÄmÅmÆmÇmÈmÉmÊmËmÌmÍvÎvÏvÐvÑvÒvÓvÔvÕvÖv×vØvÙvÚvÛ"
	optimizer
,
>serving_default"
signature_map
<:: 2"auto_encoder/encoder/conv2d/kernel
.:, 2 auto_encoder/encoder/conv2d/bias
>:<  2$auto_encoder/encoder/conv2d_1/kernel
0:. 2"auto_encoder/encoder/conv2d_1/bias
>:< 2$auto_encoder/encoder/conv2d_2/kernel
0:.2"auto_encoder/encoder/conv2d_2/bias
>:<2$auto_encoder/decoder/conv2d_3/kernel
0:.2"auto_encoder/decoder/conv2d_3/bias
>:< 2$auto_encoder/decoder/conv2d_4/kernel
0:. 2"auto_encoder/decoder/conv2d_4/bias
>:<  2$auto_encoder/decoder/conv2d_5/kernel
0:. 2"auto_encoder/decoder/conv2d_5/bias
>:< 2$auto_encoder/decoder/conv2d_6/kernel
0:.2"auto_encoder/decoder/conv2d_6/bias
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
éBæ
,__inference_auto_encoder_layer_call_fn_84774input_1"ª
¡²
FullArgSpec%
args
jself
jinput_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ðBí
,__inference_auto_encoder_layer_call_fn_84955input_features"ª
¡²
FullArgSpec%
args
jself
jinput_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_auto_encoder_layer_call_and_return_conditional_losses_85023input_features"ª
¡²
FullArgSpec%
args
jself
jinput_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
G__inference_auto_encoder_layer_call_and_return_conditional_losses_84881input_1"ª
¡²
FullArgSpec%
args
jself
jinput_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
­
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
ó
Etrace_02Ö
'__inference_encoder_layer_call_fn_85040ª
¡²
FullArgSpec%
args
jself
jinput_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zEtrace_0

Ftrace_02ñ
B__inference_encoder_layer_call_and_return_conditional_losses_85068ª
¡²
FullArgSpec%
args
jself
jinput_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zFtrace_0
Ý
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

kernel
bias
 M_jit_compiled_convolution_op"
_tf_keras_layer
Ý
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

kernel
bias
 T_jit_compiled_convolution_op"
_tf_keras_layer
Ý
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses

kernel
bias
 [_jit_compiled_convolution_op"
_tf_keras_layer
¥
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
­
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
ì
gtrace_02Ï
'__inference_decoder_layer_call_fn_85089£
²
FullArgSpec
args
jself
	jencoded
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zgtrace_0

htrace_02ê
B__inference_decoder_layer_call_and_return_conditional_losses_85133£
²
FullArgSpec
args
jself
	jencoded
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zhtrace_0
Ý
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

kernel
bias
 o_jit_compiled_convolution_op"
_tf_keras_layer
Ý
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses

kernel
bias
 v_jit_compiled_convolution_op"
_tf_keras_layer
Ý
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

kernel
bias
 }_jit_compiled_convolution_op"
_tf_keras_layer
â
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

kernel
bias
!_jit_compiled_convolution_op"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÊBÇ
#__inference_signature_wrapper_84922input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
 "
trackable_list_wrapper
<
*0
+1
,2
-3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ëBè
'__inference_encoder_layer_call_fn_85040input_features"ª
¡²
FullArgSpec%
args
jself
jinput_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
B__inference_encoder_layer_call_and_return_conditional_losses_85068input_features"ª
¡²
FullArgSpec%
args
jself
jinput_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
 metrics
 ¡layer_regularization_losses
¢layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
ó
£trace_02Ô
-__inference_max_pooling2d_layer_call_fn_85138¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z£trace_0

¤trace_02ï
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_85143¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¤trace_0
 "
trackable_list_wrapper
C
40
51
62
73
84"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
'__inference_decoder_layer_call_fn_85089encoded"£
²
FullArgSpec
args
jself
	jencoded
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
B__inference_decoder_layer_call_and_return_conditional_losses_85133encoded"£
²
FullArgSpec
args
jself
	jencoded
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¥non_trainable_variables
¦layers
§metrics
 ¨layer_regularization_losses
©layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ªnon_trainable_variables
«layers
¬metrics
 ­layer_regularization_losses
®layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¶
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¹non_trainable_variables
ºlayers
»metrics
 ¼layer_regularization_losses
½layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ó
¾trace_02Ô
-__inference_up_sampling2d_layer_call_fn_85148¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¾trace_0

¿trace_02ï
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_85160¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¿trace_0
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
áBÞ
-__inference_max_pooling2d_layer_call_fn_85138inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
üBù
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_85143inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
áBÞ
-__inference_up_sampling2d_layer_call_fn_85148inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
üBù
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_85160inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
A:? 2)Adam/auto_encoder/encoder/conv2d/kernel/m
3:1 2'Adam/auto_encoder/encoder/conv2d/bias/m
C:A  2+Adam/auto_encoder/encoder/conv2d_1/kernel/m
5:3 2)Adam/auto_encoder/encoder/conv2d_1/bias/m
C:A 2+Adam/auto_encoder/encoder/conv2d_2/kernel/m
5:32)Adam/auto_encoder/encoder/conv2d_2/bias/m
C:A2+Adam/auto_encoder/decoder/conv2d_3/kernel/m
5:32)Adam/auto_encoder/decoder/conv2d_3/bias/m
C:A 2+Adam/auto_encoder/decoder/conv2d_4/kernel/m
5:3 2)Adam/auto_encoder/decoder/conv2d_4/bias/m
C:A  2+Adam/auto_encoder/decoder/conv2d_5/kernel/m
5:3 2)Adam/auto_encoder/decoder/conv2d_5/bias/m
C:A 2+Adam/auto_encoder/decoder/conv2d_6/kernel/m
5:32)Adam/auto_encoder/decoder/conv2d_6/bias/m
A:? 2)Adam/auto_encoder/encoder/conv2d/kernel/v
3:1 2'Adam/auto_encoder/encoder/conv2d/bias/v
C:A  2+Adam/auto_encoder/encoder/conv2d_1/kernel/v
5:3 2)Adam/auto_encoder/encoder/conv2d_1/bias/v
C:A 2+Adam/auto_encoder/encoder/conv2d_2/kernel/v
5:32)Adam/auto_encoder/encoder/conv2d_2/bias/v
C:A2+Adam/auto_encoder/decoder/conv2d_3/kernel/v
5:32)Adam/auto_encoder/decoder/conv2d_3/bias/v
C:A 2+Adam/auto_encoder/decoder/conv2d_4/kernel/v
5:3 2)Adam/auto_encoder/decoder/conv2d_4/bias/v
C:A  2+Adam/auto_encoder/decoder/conv2d_5/kernel/v
5:3 2)Adam/auto_encoder/decoder/conv2d_5/bias/v
C:A 2+Adam/auto_encoder/decoder/conv2d_6/kernel/v
5:32)Adam/auto_encoder/decoder/conv2d_6/bias/v¬
 __inference__wrapped_model_846008¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
ª ";ª8
6
output_1*'
output_1ÿÿÿÿÿÿÿÿÿÄ
G__inference_auto_encoder_layer_call_and_return_conditional_losses_84881y8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ì
G__inference_auto_encoder_layer_call_and_return_conditional_losses_85023?¢<
5¢2
0-
input_featuresÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
,__inference_auto_encoder_layer_call_fn_84774l8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ£
,__inference_auto_encoder_layer_call_fn_84955s?¢<
5¢2
0-
input_featuresÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ¹
B__inference_decoder_layer_call_and_return_conditional_losses_85133s8¢5
.¢+
)&
encodedÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
'__inference_decoder_layer_call_fn_85089f8¢5
.¢+
)&
encodedÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ¾
B__inference_encoder_layer_call_and_return_conditional_losses_85068x?¢<
5¢2
0-
input_featuresÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
'__inference_encoder_layer_call_fn_85040k?¢<
5¢2
0-
input_featuresÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿë
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_85143R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_max_pooling2d_layer_call_fn_85138R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
#__inference_signature_wrapper_84922C¢@
¢ 
9ª6
4
input_1)&
input_1ÿÿÿÿÿÿÿÿÿ";ª8
6
output_1*'
output_1ÿÿÿÿÿÿÿÿÿë
H__inference_up_sampling2d_layer_call_and_return_conditional_losses_85160R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_up_sampling2d_layer_call_fn_85148R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ