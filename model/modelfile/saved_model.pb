??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
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
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.02unknown8??
?
text_cnn_1/embedding/embeddingsVarHandleOp*
shape:
??*0
shared_name!text_cnn_1/embedding/embeddings*
dtype0*
_output_shapes
: 
?
3text_cnn_1/embedding/embeddings/Read/ReadVariableOpReadVariableOptext_cnn_1/embedding/embeddings*
dtype0* 
_output_shapes
:
??
?
text_cnn_1/dense/kernelVarHandleOp*
shape:	?*(
shared_nametext_cnn_1/dense/kernel*
dtype0*
_output_shapes
: 
?
+text_cnn_1/dense/kernel/Read/ReadVariableOpReadVariableOptext_cnn_1/dense/kernel*
dtype0*
_output_shapes
:	?
?
text_cnn_1/dense/biasVarHandleOp*
shape:*&
shared_nametext_cnn_1/dense/bias*
dtype0*
_output_shapes
: 
{
)text_cnn_1/dense/bias/Read/ReadVariableOpReadVariableOptext_cnn_1/dense/bias*
dtype0*
_output_shapes
:
`
beta_1VarHandleOp*
shape: *
shared_namebeta_1*
dtype0*
_output_shapes
: 
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
dtype0*
_output_shapes
: 
`
beta_2VarHandleOp*
shape: *
shared_namebeta_2*
dtype0*
_output_shapes
: 
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
dtype0*
_output_shapes
: 
^
decayVarHandleOp*
shape: *
shared_namedecay*
dtype0*
_output_shapes
: 
W
decay/Read/ReadVariableOpReadVariableOpdecay*
dtype0*
_output_shapes
: 
n
learning_rateVarHandleOp*
shape: *
shared_namelearning_rate*
dtype0*
_output_shapes
: 
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
dtype0*
_output_shapes
: 
f
	Adam/iterVarHandleOp*
shape: *
shared_name	Adam/iter*
dtype0	*
_output_shapes
: 
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
?
text_cnn_1/conv1d/kernelVarHandleOp*
shape:?*)
shared_nametext_cnn_1/conv1d/kernel*
dtype0*
_output_shapes
: 
?
,text_cnn_1/conv1d/kernel/Read/ReadVariableOpReadVariableOptext_cnn_1/conv1d/kernel*
dtype0*#
_output_shapes
:?
?
text_cnn_1/conv1d/biasVarHandleOp*
shape:?*'
shared_nametext_cnn_1/conv1d/bias*
dtype0*
_output_shapes
: 
~
*text_cnn_1/conv1d/bias/Read/ReadVariableOpReadVariableOptext_cnn_1/conv1d/bias*
dtype0*
_output_shapes	
:?
?
text_cnn_1/conv1d_1/kernelVarHandleOp*
shape:?*+
shared_nametext_cnn_1/conv1d_1/kernel*
dtype0*
_output_shapes
: 
?
.text_cnn_1/conv1d_1/kernel/Read/ReadVariableOpReadVariableOptext_cnn_1/conv1d_1/kernel*
dtype0*#
_output_shapes
:?
?
text_cnn_1/conv1d_1/biasVarHandleOp*
shape:?*)
shared_nametext_cnn_1/conv1d_1/bias*
dtype0*
_output_shapes
: 
?
,text_cnn_1/conv1d_1/bias/Read/ReadVariableOpReadVariableOptext_cnn_1/conv1d_1/bias*
dtype0*
_output_shapes	
:?
?
text_cnn_1/conv1d_2/kernelVarHandleOp*
shape:?*+
shared_nametext_cnn_1/conv1d_2/kernel*
dtype0*
_output_shapes
: 
?
.text_cnn_1/conv1d_2/kernel/Read/ReadVariableOpReadVariableOptext_cnn_1/conv1d_2/kernel*
dtype0*#
_output_shapes
:?
?
text_cnn_1/conv1d_2/biasVarHandleOp*
shape:?*)
shared_nametext_cnn_1/conv1d_2/bias*
dtype0*
_output_shapes
: 
?
,text_cnn_1/conv1d_2/bias/Read/ReadVariableOpReadVariableOptext_cnn_1/conv1d_2/bias*
dtype0*
_output_shapes	
:?
^
totalVarHandleOp*
shape: *
shared_nametotal*
dtype0*
_output_shapes
: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shape: *
shared_namecount*
dtype0*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
?
&Adam/text_cnn_1/embedding/embeddings/mVarHandleOp*
shape:
??*7
shared_name(&Adam/text_cnn_1/embedding/embeddings/m*
dtype0*
_output_shapes
: 
?
:Adam/text_cnn_1/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp&Adam/text_cnn_1/embedding/embeddings/m*
dtype0* 
_output_shapes
:
??
?
Adam/text_cnn_1/dense/kernel/mVarHandleOp*
shape:	?*/
shared_name Adam/text_cnn_1/dense/kernel/m*
dtype0*
_output_shapes
: 
?
2Adam/text_cnn_1/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/text_cnn_1/dense/kernel/m*
dtype0*
_output_shapes
:	?
?
Adam/text_cnn_1/dense/bias/mVarHandleOp*
shape:*-
shared_nameAdam/text_cnn_1/dense/bias/m*
dtype0*
_output_shapes
: 
?
0Adam/text_cnn_1/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/text_cnn_1/dense/bias/m*
dtype0*
_output_shapes
:
?
Adam/text_cnn_1/conv1d/kernel/mVarHandleOp*
shape:?*0
shared_name!Adam/text_cnn_1/conv1d/kernel/m*
dtype0*
_output_shapes
: 
?
3Adam/text_cnn_1/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/text_cnn_1/conv1d/kernel/m*
dtype0*#
_output_shapes
:?
?
Adam/text_cnn_1/conv1d/bias/mVarHandleOp*
shape:?*.
shared_nameAdam/text_cnn_1/conv1d/bias/m*
dtype0*
_output_shapes
: 
?
1Adam/text_cnn_1/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/text_cnn_1/conv1d/bias/m*
dtype0*
_output_shapes	
:?
?
!Adam/text_cnn_1/conv1d_1/kernel/mVarHandleOp*
shape:?*2
shared_name#!Adam/text_cnn_1/conv1d_1/kernel/m*
dtype0*
_output_shapes
: 
?
5Adam/text_cnn_1/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/text_cnn_1/conv1d_1/kernel/m*
dtype0*#
_output_shapes
:?
?
Adam/text_cnn_1/conv1d_1/bias/mVarHandleOp*
shape:?*0
shared_name!Adam/text_cnn_1/conv1d_1/bias/m*
dtype0*
_output_shapes
: 
?
3Adam/text_cnn_1/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/text_cnn_1/conv1d_1/bias/m*
dtype0*
_output_shapes	
:?
?
!Adam/text_cnn_1/conv1d_2/kernel/mVarHandleOp*
shape:?*2
shared_name#!Adam/text_cnn_1/conv1d_2/kernel/m*
dtype0*
_output_shapes
: 
?
5Adam/text_cnn_1/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/text_cnn_1/conv1d_2/kernel/m*
dtype0*#
_output_shapes
:?
?
Adam/text_cnn_1/conv1d_2/bias/mVarHandleOp*
shape:?*0
shared_name!Adam/text_cnn_1/conv1d_2/bias/m*
dtype0*
_output_shapes
: 
?
3Adam/text_cnn_1/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/text_cnn_1/conv1d_2/bias/m*
dtype0*
_output_shapes	
:?
?
&Adam/text_cnn_1/embedding/embeddings/vVarHandleOp*
shape:
??*7
shared_name(&Adam/text_cnn_1/embedding/embeddings/v*
dtype0*
_output_shapes
: 
?
:Adam/text_cnn_1/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp&Adam/text_cnn_1/embedding/embeddings/v*
dtype0* 
_output_shapes
:
??
?
Adam/text_cnn_1/dense/kernel/vVarHandleOp*
shape:	?*/
shared_name Adam/text_cnn_1/dense/kernel/v*
dtype0*
_output_shapes
: 
?
2Adam/text_cnn_1/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/text_cnn_1/dense/kernel/v*
dtype0*
_output_shapes
:	?
?
Adam/text_cnn_1/dense/bias/vVarHandleOp*
shape:*-
shared_nameAdam/text_cnn_1/dense/bias/v*
dtype0*
_output_shapes
: 
?
0Adam/text_cnn_1/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/text_cnn_1/dense/bias/v*
dtype0*
_output_shapes
:
?
Adam/text_cnn_1/conv1d/kernel/vVarHandleOp*
shape:?*0
shared_name!Adam/text_cnn_1/conv1d/kernel/v*
dtype0*
_output_shapes
: 
?
3Adam/text_cnn_1/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/text_cnn_1/conv1d/kernel/v*
dtype0*#
_output_shapes
:?
?
Adam/text_cnn_1/conv1d/bias/vVarHandleOp*
shape:?*.
shared_nameAdam/text_cnn_1/conv1d/bias/v*
dtype0*
_output_shapes
: 
?
1Adam/text_cnn_1/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/text_cnn_1/conv1d/bias/v*
dtype0*
_output_shapes	
:?
?
!Adam/text_cnn_1/conv1d_1/kernel/vVarHandleOp*
shape:?*2
shared_name#!Adam/text_cnn_1/conv1d_1/kernel/v*
dtype0*
_output_shapes
: 
?
5Adam/text_cnn_1/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/text_cnn_1/conv1d_1/kernel/v*
dtype0*#
_output_shapes
:?
?
Adam/text_cnn_1/conv1d_1/bias/vVarHandleOp*
shape:?*0
shared_name!Adam/text_cnn_1/conv1d_1/bias/v*
dtype0*
_output_shapes
: 
?
3Adam/text_cnn_1/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/text_cnn_1/conv1d_1/bias/v*
dtype0*
_output_shapes	
:?
?
!Adam/text_cnn_1/conv1d_2/kernel/vVarHandleOp*
shape:?*2
shared_name#!Adam/text_cnn_1/conv1d_2/kernel/v*
dtype0*
_output_shapes
: 
?
5Adam/text_cnn_1/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/text_cnn_1/conv1d_2/kernel/v*
dtype0*#
_output_shapes
:?
?
Adam/text_cnn_1/conv1d_2/bias/vVarHandleOp*
shape:?*0
shared_name!Adam/text_cnn_1/conv1d_2/bias/v*
dtype0*
_output_shapes
: 
?
3Adam/text_cnn_1/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/text_cnn_1/conv1d_2/bias/v*
dtype0*
_output_shapes	
:?

NoOpNoOp
?7
ConstConst"/device:CPU:0*?6
value?6B?6 B?6
?
kernel_sizes
	embedding
	convs
max_poolings

classifier
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
 
b

embeddings
	variables
regularization_losses
trainable_variables
	keras_api

0
1
2

0
1
2
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
?

beta_1

beta_2
	decay
 learning_rate
!itermpmqmr"ms#mt$mu%mv&mw'mxvyvzv{"v|#v}$v~%v&v?'v?
?
0
"1
#2
$3
%4
&5
'6
7
8
 
?
0
"1
#2
$3
%4
&5
'6
7
8
?
	variables
(non_trainable_variables
regularization_losses
)layer_regularization_losses

*layers
+metrics
	trainable_variables
 
db
VARIABLE_VALUEtext_cnn_1/embedding/embeddings/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
?
	variables
,non_trainable_variables
regularization_losses
-layer_regularization_losses

.layers
/metrics
trainable_variables
h

"kernel
#bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
h

$kernel
%bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
h

&kernel
'bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
R
<	variables
=regularization_losses
>trainable_variables
?	keras_api
R
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
R
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
YW
VARIABLE_VALUEtext_cnn_1/dense/kernel,classifier/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtext_cnn_1/dense/bias*classifier/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
	variables
Hnon_trainable_variables
regularization_losses
Ilayer_regularization_losses

Jlayers
Kmetrics
trainable_variables
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEtext_cnn_1/conv1d/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEtext_cnn_1/conv1d/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEtext_cnn_1/conv1d_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEtext_cnn_1/conv1d_1/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEtext_cnn_1/conv1d_2/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEtext_cnn_1/conv1d_2/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
 
 
8
0
1
2
3
4
5
6
7

L0
 
 
 
 

"0
#1
 

"0
#1
?
0	variables
Mnon_trainable_variables
1regularization_losses
Nlayer_regularization_losses

Olayers
Pmetrics
2trainable_variables

$0
%1
 

$0
%1
?
4	variables
Qnon_trainable_variables
5regularization_losses
Rlayer_regularization_losses

Slayers
Tmetrics
6trainable_variables

&0
'1
 

&0
'1
?
8	variables
Unon_trainable_variables
9regularization_losses
Vlayer_regularization_losses

Wlayers
Xmetrics
:trainable_variables
 
 
 
?
<	variables
Ynon_trainable_variables
=regularization_losses
Zlayer_regularization_losses

[layers
\metrics
>trainable_variables
 
 
 
?
@	variables
]non_trainable_variables
Aregularization_losses
^layer_regularization_losses

_layers
`metrics
Btrainable_variables
 
 
 
?
D	variables
anon_trainable_variables
Eregularization_losses
blayer_regularization_losses

clayers
dmetrics
Ftrainable_variables
 
 
 
 
x
	etotal
	fcount
g
_fn_kwargs
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1
 
 
?
h	variables
lnon_trainable_variables
iregularization_losses
mlayer_regularization_losses

nlayers
ometrics
jtrainable_variables

e0
f1
 
 
 
??
VARIABLE_VALUE&Adam/text_cnn_1/embedding/embeddings/mKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/text_cnn_1/dense/kernel/mHclassifier/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/text_cnn_1/dense/bias/mFclassifier/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/text_cnn_1/conv1d/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/text_cnn_1/conv1d/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/text_cnn_1/conv1d_1/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/text_cnn_1/conv1d_1/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/text_cnn_1/conv1d_2/kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/text_cnn_1/conv1d_2/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/text_cnn_1/embedding/embeddings/vKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/text_cnn_1/dense/kernel/vHclassifier/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/text_cnn_1/dense/bias/vFclassifier/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/text_cnn_1/conv1d/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/text_cnn_1/conv1d/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/text_cnn_1/conv1d_1/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/text_cnn_1/conv1d_1/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/text_cnn_1/conv1d_2/kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/text_cnn_1/conv1d_2/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
z
serving_default_input_1Placeholder*
shape:?????????d*
dtype0*'
_output_shapes
:?????????d
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1text_cnn_1/embedding/embeddingstext_cnn_1/conv1d/kerneltext_cnn_1/conv1d/biastext_cnn_1/conv1d_1/kerneltext_cnn_1/conv1d_1/biastext_cnn_1/conv1d_2/kerneltext_cnn_1/conv1d_2/biastext_cnn_1/dense/kerneltext_cnn_1/dense/bias*,
_gradient_op_typePartitionedCall-23614*,
f'R%
#__inference_signature_wrapper_23506*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2
*'
_output_shapes
:?????????
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename3text_cnn_1/embedding/embeddings/Read/ReadVariableOp+text_cnn_1/dense/kernel/Read/ReadVariableOp)text_cnn_1/dense/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOp,text_cnn_1/conv1d/kernel/Read/ReadVariableOp*text_cnn_1/conv1d/bias/Read/ReadVariableOp.text_cnn_1/conv1d_1/kernel/Read/ReadVariableOp,text_cnn_1/conv1d_1/bias/Read/ReadVariableOp.text_cnn_1/conv1d_2/kernel/Read/ReadVariableOp,text_cnn_1/conv1d_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp:Adam/text_cnn_1/embedding/embeddings/m/Read/ReadVariableOp2Adam/text_cnn_1/dense/kernel/m/Read/ReadVariableOp0Adam/text_cnn_1/dense/bias/m/Read/ReadVariableOp3Adam/text_cnn_1/conv1d/kernel/m/Read/ReadVariableOp1Adam/text_cnn_1/conv1d/bias/m/Read/ReadVariableOp5Adam/text_cnn_1/conv1d_1/kernel/m/Read/ReadVariableOp3Adam/text_cnn_1/conv1d_1/bias/m/Read/ReadVariableOp5Adam/text_cnn_1/conv1d_2/kernel/m/Read/ReadVariableOp3Adam/text_cnn_1/conv1d_2/bias/m/Read/ReadVariableOp:Adam/text_cnn_1/embedding/embeddings/v/Read/ReadVariableOp2Adam/text_cnn_1/dense/kernel/v/Read/ReadVariableOp0Adam/text_cnn_1/dense/bias/v/Read/ReadVariableOp3Adam/text_cnn_1/conv1d/kernel/v/Read/ReadVariableOp1Adam/text_cnn_1/conv1d/bias/v/Read/ReadVariableOp5Adam/text_cnn_1/conv1d_1/kernel/v/Read/ReadVariableOp3Adam/text_cnn_1/conv1d_1/bias/v/Read/ReadVariableOp5Adam/text_cnn_1/conv1d_2/kernel/v/Read/ReadVariableOp3Adam/text_cnn_1/conv1d_2/bias/v/Read/ReadVariableOpConst*,
_gradient_op_typePartitionedCall-23670*'
f"R 
__inference__traced_save_23669*
Tout
2**
config_proto

CPU

GPU 2J 8*/
Tin(
&2$	*
_output_shapes
: 
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametext_cnn_1/embedding/embeddingstext_cnn_1/dense/kerneltext_cnn_1/dense/biasbeta_1beta_2decaylearning_rate	Adam/itertext_cnn_1/conv1d/kerneltext_cnn_1/conv1d/biastext_cnn_1/conv1d_1/kerneltext_cnn_1/conv1d_1/biastext_cnn_1/conv1d_2/kerneltext_cnn_1/conv1d_2/biastotalcount&Adam/text_cnn_1/embedding/embeddings/mAdam/text_cnn_1/dense/kernel/mAdam/text_cnn_1/dense/bias/mAdam/text_cnn_1/conv1d/kernel/mAdam/text_cnn_1/conv1d/bias/m!Adam/text_cnn_1/conv1d_1/kernel/mAdam/text_cnn_1/conv1d_1/bias/m!Adam/text_cnn_1/conv1d_2/kernel/mAdam/text_cnn_1/conv1d_2/bias/m&Adam/text_cnn_1/embedding/embeddings/vAdam/text_cnn_1/dense/kernel/vAdam/text_cnn_1/dense/bias/vAdam/text_cnn_1/conv1d/kernel/vAdam/text_cnn_1/conv1d/bias/v!Adam/text_cnn_1/conv1d_1/kernel/vAdam/text_cnn_1/conv1d_1/bias/v!Adam/text_cnn_1/conv1d_2/kernel/vAdam/text_cnn_1/conv1d_2/bias/v*,
_gradient_op_typePartitionedCall-23785**
f%R#
!__inference__traced_restore_23784*
Tout
2**
config_proto

CPU

GPU 2J 8*.
Tin'
%2#*
_output_shapes
: ??
?
?
(__inference_conv1d_2_layer_call_fn_23327

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-23322*L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_23316*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*5
_output_shapes#
!:????????????????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_23535

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
(__inference_text_cnn_layer_call_fn_23484
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*,
_gradient_op_typePartitionedCall-23472*L
fGRE
C__inference_text_cnn_layer_call_and_return_conditional_losses_23466*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2
*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*J
_input_shapes9
7:?????????d:::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :' #
!
_user_specified_name	input_1: : :	 : 
?Z
?
 __inference__wrapped_model_23237
input_1D
@text_cnn_embedding_embedding_lookup_read_readvariableop_resource?
;text_cnn_conv1d_conv1d_expanddims_1_readvariableop_resource3
/text_cnn_conv1d_biasadd_readvariableop_resourceA
=text_cnn_conv1d_1_conv1d_expanddims_1_readvariableop_resource5
1text_cnn_conv1d_1_biasadd_readvariableop_resourceA
=text_cnn_conv1d_2_conv1d_expanddims_1_readvariableop_resource5
1text_cnn_conv1d_2_biasadd_readvariableop_resource1
-text_cnn_dense_matmul_readvariableop_resource2
.text_cnn_dense_biasadd_readvariableop_resource
identity??&text_cnn/conv1d/BiasAdd/ReadVariableOp?2text_cnn/conv1d/conv1d/ExpandDims_1/ReadVariableOp?(text_cnn/conv1d_1/BiasAdd/ReadVariableOp?4text_cnn/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?(text_cnn/conv1d_2/BiasAdd/ReadVariableOp?4text_cnn/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?%text_cnn/dense/BiasAdd/ReadVariableOp?$text_cnn/dense/MatMul/ReadVariableOp?#text_cnn/embedding/embedding_lookup?7text_cnn/embedding/embedding_lookup/Read/ReadVariableOp?
7text_cnn/embedding/embedding_lookup/Read/ReadVariableOpReadVariableOp@text_cnn_embedding_embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
???
,text_cnn/embedding/embedding_lookup/IdentityIdentity?text_cnn/embedding/embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
#text_cnn/embedding/embedding_lookupResourceGather@text_cnn_embedding_embedding_lookup_read_readvariableop_resourceinput_18^text_cnn/embedding/embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*J
_class@
><loc:@text_cnn/embedding/embedding_lookup/Read/ReadVariableOp*
dtype0*+
_output_shapes
:?????????d?
.text_cnn/embedding/embedding_lookup/Identity_1Identity,text_cnn/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*J
_class@
><loc:@text_cnn/embedding/embedding_lookup/Read/ReadVariableOp*+
_output_shapes
:?????????d?
.text_cnn/embedding/embedding_lookup/Identity_2Identity7text_cnn/embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????dg
%text_cnn/conv1d/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: ?
!text_cnn/conv1d/conv1d/ExpandDims
ExpandDims7text_cnn/embedding/embedding_lookup/Identity_2:output:0.text_cnn/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d?
2text_cnn/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;text_cnn_conv1d_conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:?i
'text_cnn/conv1d/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: ?
#text_cnn/conv1d/conv1d/ExpandDims_1
ExpandDims:text_cnn/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:00text_cnn/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:??
text_cnn/conv1d/conv1dConv2D*text_cnn/conv1d/conv1d/ExpandDims:output:0,text_cnn/conv1d/conv1d/ExpandDims_1:output:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:?????????a??
text_cnn/conv1d/conv1d/SqueezeSqueezetext_cnn/conv1d/conv1d:output:0*
squeeze_dims
*
T0*,
_output_shapes
:?????????a??
&text_cnn/conv1d/BiasAdd/ReadVariableOpReadVariableOp/text_cnn_conv1d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
text_cnn/conv1d/BiasAddBiasAdd'text_cnn/conv1d/conv1d/Squeeze:output:0.text_cnn/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????a?u
text_cnn/conv1d/ReluRelu text_cnn/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????a?u
3text_cnn/global_max_pooling1d/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: ?
!text_cnn/global_max_pooling1d/MaxMax"text_cnn/conv1d/Relu:activations:0<text_cnn/global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:??????????i
'text_cnn/conv1d_1/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: ?
#text_cnn/conv1d_1/conv1d/ExpandDims
ExpandDims7text_cnn/embedding/embedding_lookup/Identity_2:output:00text_cnn/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d?
4text_cnn/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=text_cnn_conv1d_1_conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:?k
)text_cnn/conv1d_1/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: ?
%text_cnn/conv1d_1/conv1d/ExpandDims_1
ExpandDims<text_cnn/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:02text_cnn/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:??
text_cnn/conv1d_1/conv1dConv2D,text_cnn/conv1d_1/conv1d/ExpandDims:output:0.text_cnn/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:?????????`??
 text_cnn/conv1d_1/conv1d/SqueezeSqueeze!text_cnn/conv1d_1/conv1d:output:0*
squeeze_dims
*
T0*,
_output_shapes
:?????????`??
(text_cnn/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp1text_cnn_conv1d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
text_cnn/conv1d_1/BiasAddBiasAdd)text_cnn/conv1d_1/conv1d/Squeeze:output:00text_cnn/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????`?y
text_cnn/conv1d_1/ReluRelu"text_cnn/conv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:?????????`?w
5text_cnn/global_max_pooling1d_1/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: ?
#text_cnn/global_max_pooling1d_1/MaxMax$text_cnn/conv1d_1/Relu:activations:0>text_cnn/global_max_pooling1d_1/Max/reduction_indices:output:0*
T0*(
_output_shapes
:??????????i
'text_cnn/conv1d_2/conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: ?
#text_cnn/conv1d_2/conv1d/ExpandDims
ExpandDims7text_cnn/embedding/embedding_lookup/Identity_2:output:00text_cnn/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????d?
4text_cnn/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp=text_cnn_conv1d_2_conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:?k
)text_cnn/conv1d_2/conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: ?
%text_cnn/conv1d_2/conv1d/ExpandDims_1
ExpandDims<text_cnn/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:02text_cnn/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:??
text_cnn/conv1d_2/conv1dConv2D,text_cnn/conv1d_2/conv1d/ExpandDims:output:0.text_cnn/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:?????????_??
 text_cnn/conv1d_2/conv1d/SqueezeSqueeze!text_cnn/conv1d_2/conv1d:output:0*
squeeze_dims
*
T0*,
_output_shapes
:?????????_??
(text_cnn/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp1text_cnn_conv1d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
text_cnn/conv1d_2/BiasAddBiasAdd)text_cnn/conv1d_2/conv1d/Squeeze:output:00text_cnn/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????_?y
text_cnn/conv1d_2/ReluRelu"text_cnn/conv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:?????????_?w
5text_cnn/global_max_pooling1d_2/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: ?
#text_cnn/global_max_pooling1d_2/MaxMax$text_cnn/conv1d_2/Relu:activations:0>text_cnn/global_max_pooling1d_2/Max/reduction_indices:output:0*
T0*(
_output_shapes
:??????????b
 text_cnn/concatenate/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
text_cnn/concatenate/concatConcatV2*text_cnn/global_max_pooling1d/Max:output:0,text_cnn/global_max_pooling1d_1/Max:output:0,text_cnn/global_max_pooling1d_2/Max:output:0)text_cnn/concatenate/concat/axis:output:0*
T0*
N*(
_output_shapes
:??????????~
text_cnn/dropout/IdentityIdentity$text_cnn/concatenate/concat:output:0*
T0*(
_output_shapes
:???????????
$text_cnn/dense/MatMul/ReadVariableOpReadVariableOp-text_cnn_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	??
text_cnn/dense/MatMulMatMul"text_cnn/dropout/Identity:output:0,text_cnn/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
%text_cnn/dense/BiasAdd/ReadVariableOpReadVariableOp.text_cnn_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
text_cnn/dense/BiasAddBiasAddtext_cnn/dense/MatMul:product:0-text_cnn/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
text_cnn/dense/SigmoidSigmoidtext_cnn/dense/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitytext_cnn/dense/Sigmoid:y:0'^text_cnn/conv1d/BiasAdd/ReadVariableOp3^text_cnn/conv1d/conv1d/ExpandDims_1/ReadVariableOp)^text_cnn/conv1d_1/BiasAdd/ReadVariableOp5^text_cnn/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp)^text_cnn/conv1d_2/BiasAdd/ReadVariableOp5^text_cnn/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp&^text_cnn/dense/BiasAdd/ReadVariableOp%^text_cnn/dense/MatMul/ReadVariableOp$^text_cnn/embedding/embedding_lookup8^text_cnn/embedding/embedding_lookup/Read/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*J
_input_shapes9
7:?????????d:::::::::2l
4text_cnn/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp4text_cnn/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2N
%text_cnn/dense/BiasAdd/ReadVariableOp%text_cnn/dense/BiasAdd/ReadVariableOp2l
4text_cnn/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp4text_cnn/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2P
&text_cnn/conv1d/BiasAdd/ReadVariableOp&text_cnn/conv1d/BiasAdd/ReadVariableOp2T
(text_cnn/conv1d_2/BiasAdd/ReadVariableOp(text_cnn/conv1d_2/BiasAdd/ReadVariableOp2r
7text_cnn/embedding/embedding_lookup/Read/ReadVariableOp7text_cnn/embedding/embedding_lookup/Read/ReadVariableOp2T
(text_cnn/conv1d_1/BiasAdd/ReadVariableOp(text_cnn/conv1d_1/BiasAdd/ReadVariableOp2L
$text_cnn/dense/MatMul/ReadVariableOp$text_cnn/dense/MatMul/ReadVariableOp2J
#text_cnn/embedding/embedding_lookup#text_cnn/embedding/embedding_lookup2h
2text_cnn/conv1d/conv1d/ExpandDims_1/ReadVariableOp2text_cnn/conv1d/conv1d/ExpandDims_1/ReadVariableOp: : : : : :' #
!
_user_specified_name	input_1: : :	 : 
?
R
6__inference_global_max_pooling1d_2_layer_call_fn_23381

inputs
identity?
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-23378*Z
fURS
Q__inference_global_max_pooling1d_2_layer_call_and_return_conditional_losses_23372*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:??????????????????i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:& "
 
_user_specified_nameinputs
?
P
4__inference_global_max_pooling1d_layer_call_fn_23345

inputs
identity?
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-23342*X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_23336*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:??????????????????i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:& "
 
_user_specified_nameinputs
?E
?
__inference__traced_save_23669
file_prefix>
:savev2_text_cnn_1_embedding_embeddings_read_readvariableop6
2savev2_text_cnn_1_dense_kernel_read_readvariableop4
0savev2_text_cnn_1_dense_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	7
3savev2_text_cnn_1_conv1d_kernel_read_readvariableop5
1savev2_text_cnn_1_conv1d_bias_read_readvariableop9
5savev2_text_cnn_1_conv1d_1_kernel_read_readvariableop7
3savev2_text_cnn_1_conv1d_1_bias_read_readvariableop9
5savev2_text_cnn_1_conv1d_2_kernel_read_readvariableop7
3savev2_text_cnn_1_conv1d_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopE
Asavev2_adam_text_cnn_1_embedding_embeddings_m_read_readvariableop=
9savev2_adam_text_cnn_1_dense_kernel_m_read_readvariableop;
7savev2_adam_text_cnn_1_dense_bias_m_read_readvariableop>
:savev2_adam_text_cnn_1_conv1d_kernel_m_read_readvariableop<
8savev2_adam_text_cnn_1_conv1d_bias_m_read_readvariableop@
<savev2_adam_text_cnn_1_conv1d_1_kernel_m_read_readvariableop>
:savev2_adam_text_cnn_1_conv1d_1_bias_m_read_readvariableop@
<savev2_adam_text_cnn_1_conv1d_2_kernel_m_read_readvariableop>
:savev2_adam_text_cnn_1_conv1d_2_bias_m_read_readvariableopE
Asavev2_adam_text_cnn_1_embedding_embeddings_v_read_readvariableop=
9savev2_adam_text_cnn_1_dense_kernel_v_read_readvariableop;
7savev2_adam_text_cnn_1_dense_bias_v_read_readvariableop>
:savev2_adam_text_cnn_1_conv1d_kernel_v_read_readvariableop<
8savev2_adam_text_cnn_1_conv1d_bias_v_read_readvariableop@
<savev2_adam_text_cnn_1_conv1d_1_kernel_v_read_readvariableop>
:savev2_adam_text_cnn_1_conv1d_1_bias_v_read_readvariableop@
<savev2_adam_text_cnn_1_conv1d_2_kernel_v_read_readvariableop>
:savev2_adam_text_cnn_1_conv1d_2_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_05a930fb8b8c4d0d88b074315a944003/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?"B/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB,classifier/kernel/.ATTRIBUTES/VARIABLE_VALUEB*classifier/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHclassifier/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFclassifier/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHclassifier/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFclassifier/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:"?
SaveV2/shape_and_slicesConst"/device:CPU:0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:"?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0:savev2_text_cnn_1_embedding_embeddings_read_readvariableop2savev2_text_cnn_1_dense_kernel_read_readvariableop0savev2_text_cnn_1_dense_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop3savev2_text_cnn_1_conv1d_kernel_read_readvariableop1savev2_text_cnn_1_conv1d_bias_read_readvariableop5savev2_text_cnn_1_conv1d_1_kernel_read_readvariableop3savev2_text_cnn_1_conv1d_1_bias_read_readvariableop5savev2_text_cnn_1_conv1d_2_kernel_read_readvariableop3savev2_text_cnn_1_conv1d_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopAsavev2_adam_text_cnn_1_embedding_embeddings_m_read_readvariableop9savev2_adam_text_cnn_1_dense_kernel_m_read_readvariableop7savev2_adam_text_cnn_1_dense_bias_m_read_readvariableop:savev2_adam_text_cnn_1_conv1d_kernel_m_read_readvariableop8savev2_adam_text_cnn_1_conv1d_bias_m_read_readvariableop<savev2_adam_text_cnn_1_conv1d_1_kernel_m_read_readvariableop:savev2_adam_text_cnn_1_conv1d_1_bias_m_read_readvariableop<savev2_adam_text_cnn_1_conv1d_2_kernel_m_read_readvariableop:savev2_adam_text_cnn_1_conv1d_2_bias_m_read_readvariableopAsavev2_adam_text_cnn_1_embedding_embeddings_v_read_readvariableop9savev2_adam_text_cnn_1_dense_kernel_v_read_readvariableop7savev2_adam_text_cnn_1_dense_bias_v_read_readvariableop:savev2_adam_text_cnn_1_conv1d_kernel_v_read_readvariableop8savev2_adam_text_cnn_1_conv1d_bias_v_read_readvariableop<savev2_adam_text_cnn_1_conv1d_1_kernel_v_read_readvariableop:savev2_adam_text_cnn_1_conv1d_1_bias_v_read_readvariableop<savev2_adam_text_cnn_1_conv1d_2_kernel_v_read_readvariableop:savev2_adam_text_cnn_1_conv1d_2_bias_v_read_readvariableop"/device:CPU:0*0
dtypes&
$2"	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: ?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??:	?:: : : : : :?:?:?:?:?:?: : :
??:	?::?:?:?:?:?:?:
??:	?::?:?:?:?:?:?: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : : :# : : : :	 : : : : :+ '
%
_user_specified_namefile_prefix:" : : : : : : :! : : : : : : : : :  : : : : : :
 
?
m
Q__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_23354

inputs
identityW
Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:& "
 
_user_specified_nameinputs
?
?
)__inference_embedding_layer_call_fn_23524

inputs"
statefulpartitionedcall_args_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*,
_gradient_op_typePartitionedCall-23403*M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_23397*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:?????????d?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d"
identityIdentity:output:0**
_input_shapes
:?????????d:22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs
̈́
?
!__inference__traced_restore_23784
file_prefix4
0assignvariableop_text_cnn_1_embedding_embeddings.
*assignvariableop_1_text_cnn_1_dense_kernel,
(assignvariableop_2_text_cnn_1_dense_bias
assignvariableop_3_beta_1
assignvariableop_4_beta_2
assignvariableop_5_decay$
 assignvariableop_6_learning_rate 
assignvariableop_7_adam_iter/
+assignvariableop_8_text_cnn_1_conv1d_kernel-
)assignvariableop_9_text_cnn_1_conv1d_bias2
.assignvariableop_10_text_cnn_1_conv1d_1_kernel0
,assignvariableop_11_text_cnn_1_conv1d_1_bias2
.assignvariableop_12_text_cnn_1_conv1d_2_kernel0
,assignvariableop_13_text_cnn_1_conv1d_2_bias
assignvariableop_14_total
assignvariableop_15_count>
:assignvariableop_16_adam_text_cnn_1_embedding_embeddings_m6
2assignvariableop_17_adam_text_cnn_1_dense_kernel_m4
0assignvariableop_18_adam_text_cnn_1_dense_bias_m7
3assignvariableop_19_adam_text_cnn_1_conv1d_kernel_m5
1assignvariableop_20_adam_text_cnn_1_conv1d_bias_m9
5assignvariableop_21_adam_text_cnn_1_conv1d_1_kernel_m7
3assignvariableop_22_adam_text_cnn_1_conv1d_1_bias_m9
5assignvariableop_23_adam_text_cnn_1_conv1d_2_kernel_m7
3assignvariableop_24_adam_text_cnn_1_conv1d_2_bias_m>
:assignvariableop_25_adam_text_cnn_1_embedding_embeddings_v6
2assignvariableop_26_adam_text_cnn_1_dense_kernel_v4
0assignvariableop_27_adam_text_cnn_1_dense_bias_v7
3assignvariableop_28_adam_text_cnn_1_conv1d_kernel_v5
1assignvariableop_29_adam_text_cnn_1_conv1d_bias_v9
5assignvariableop_30_adam_text_cnn_1_conv1d_1_kernel_v7
3assignvariableop_31_adam_text_cnn_1_conv1d_1_bias_v9
5assignvariableop_32_adam_text_cnn_1_conv1d_2_kernel_v7
3assignvariableop_33_adam_text_cnn_1_conv1d_2_bias_v
identity_35??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?"B/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB,classifier/kernel/.ATTRIBUTES/VARIABLE_VALUEB*classifier/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHclassifier/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBFclassifier/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKembedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHclassifier/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBFclassifier/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:"?
RestoreV2/shape_and_slicesConst"/device:CPU:0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:"?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
dtypes&
$2"	*?
_output_shapes?
?::::::::::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp0assignvariableop_text_cnn_1_embedding_embeddingsIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp*assignvariableop_1_text_cnn_1_dense_kernelIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp(assignvariableop_2_text_cnn_1_dense_biasIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:y
AssignVariableOp_3AssignVariableOpassignvariableop_3_beta_1Identity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:y
AssignVariableOp_4AssignVariableOpassignvariableop_4_beta_2Identity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:x
AssignVariableOp_5AssignVariableOpassignvariableop_5_decayIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp assignvariableop_6_learning_rateIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0	*
_output_shapes
:|
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_iterIdentity_7:output:0*
dtype0	*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp+assignvariableop_8_text_cnn_1_conv1d_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp)assignvariableop_9_text_cnn_1_conv1d_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp.assignvariableop_10_text_cnn_1_conv1d_1_kernelIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp,assignvariableop_11_text_cnn_1_conv1d_1_biasIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp.assignvariableop_12_text_cnn_1_conv1d_2_kernelIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp,assignvariableop_13_text_cnn_1_conv1d_2_biasIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:{
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:{
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp:assignvariableop_16_adam_text_cnn_1_embedding_embeddings_mIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp2assignvariableop_17_adam_text_cnn_1_dense_kernel_mIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp0assignvariableop_18_adam_text_cnn_1_dense_bias_mIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp3assignvariableop_19_adam_text_cnn_1_conv1d_kernel_mIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp1assignvariableop_20_adam_text_cnn_1_conv1d_bias_mIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_text_cnn_1_conv1d_1_kernel_mIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp3assignvariableop_22_adam_text_cnn_1_conv1d_1_bias_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp5assignvariableop_23_adam_text_cnn_1_conv1d_2_kernel_mIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp3assignvariableop_24_adam_text_cnn_1_conv1d_2_bias_mIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp:assignvariableop_25_adam_text_cnn_1_embedding_embeddings_vIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp2assignvariableop_26_adam_text_cnn_1_dense_kernel_vIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp0assignvariableop_27_adam_text_cnn_1_dense_bias_vIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp3assignvariableop_28_adam_text_cnn_1_conv1d_kernel_vIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp1assignvariableop_29_adam_text_cnn_1_conv1d_bias_vIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp5assignvariableop_30_adam_text_cnn_1_conv1d_1_kernel_vIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp3assignvariableop_31_adam_text_cnn_1_conv1d_1_bias_vIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_text_cnn_1_conv1d_2_kernel_vIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp3assignvariableop_33_adam_text_cnn_1_conv1d_2_bias_vIdentity_33:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_35IdentityIdentity_34:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_35Identity_35:output:0*?
_input_shapes?
?: ::::::::::::::::::::::::::::::::::2*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_33AssignVariableOp_33: : : : : : :	 : : : : :+ '
%
_user_specified_namefile_prefix:" : : : : : : :! : : : : : : : : :  : : : : : :
 
?
m
Q__inference_global_max_pooling1d_2_layer_call_and_return_conditional_losses_23372

inputs
identityW
Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:& "
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_23542

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-23453*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_23447*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
?
(__inference_conv1d_1_layer_call_fn_23297

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-23292*L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_23286*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*5
_output_shapes#
!:????????????????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
?
A__inference_conv1d_layer_call_and_return_conditional_losses_23256

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpW
conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: ?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"???????????????????
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:?Y
conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: ?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:??
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*
strides
*
paddingVALID*9
_output_shapes'
%:#????????????????????
conv1d/SqueezeSqueezeconv1d:output:0*
squeeze_dims
*
T0*5
_output_shapes#
!:????????????????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:????????????????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:???????????????????"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
k
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_23336

inputs
identityW
Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:& "
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_23506
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9*,
_gradient_op_typePartitionedCall-23494*)
f$R"
 __inference__wrapped_model_23237*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2
*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*J
_input_shapes9
7:?????????d:::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :' #
!
_user_specified_name	input_1: : :	 : 
?
?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_23286

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpW
conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: ?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"???????????????????
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:?Y
conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: ?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:??
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*
strides
*
paddingVALID*9
_output_shapes'
%:#????????????????????
conv1d/SqueezeSqueezeconv1d:output:0*
squeeze_dims
*
T0*5
_output_shapes#
!:????????????????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:????????????????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:???????????????????"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
D__inference_embedding_layer_call_and_return_conditional_losses_23397

inputs1
-embedding_lookup_read_readvariableop_resource
identity??embedding_lookup?$embedding_lookup/Read/ReadVariableOp?
$embedding_lookup/Read/ReadVariableOpReadVariableOp-embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
??~
embedding_lookup/IdentityIdentity,embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
embedding_lookupResourceGather-embedding_lookup_read_readvariableop_resourceinputs%^embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOp*
dtype0*+
_output_shapes
:?????????d?
embedding_lookup/Identity_1Identityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOp*+
_output_shapes
:?????????d?
embedding_lookup/Identity_2Identity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????d?
IdentityIdentity$embedding_lookup/Identity_2:output:0^embedding_lookup%^embedding_lookup/Read/ReadVariableOp*
T0*+
_output_shapes
:?????????d"
identityIdentity:output:0**
_input_shapes
:?????????d:2L
$embedding_lookup/Read/ReadVariableOp$embedding_lookup/Read/ReadVariableOp2$
embedding_lookupembedding_lookup: :& "
 
_user_specified_nameinputs
?
?
&__inference_conv1d_layer_call_fn_23267

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-23262*J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_23256*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*5
_output_shapes#
!:????????????????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_23447

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
C__inference_conv1d_2_layer_call_and_return_conditional_losses_23316

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpW
conv1d/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: ?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"???????????????????
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*#
_output_shapes
:?Y
conv1d/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: ?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:??
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*
strides
*
paddingVALID*9
_output_shapes'
%:#????????????????????
conv1d/SqueezeSqueezeconv1d:output:0*
squeeze_dims
*
T0*5
_output_shapes#
!:????????????????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:??
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????^
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:????????????????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*5
_output_shapes#
!:???????????????????"
identityIdentity:output:0*;
_input_shapes*
(:??????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
R
6__inference_global_max_pooling1d_1_layer_call_fn_23363

inputs
identity?
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-23360*Z
fURS
Q__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_23354*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*0
_output_shapes
:??????????????????i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:& "
 
_user_specified_nameinputs
?
?
D__inference_embedding_layer_call_and_return_conditional_losses_23518

inputs1
-embedding_lookup_read_readvariableop_resource
identity??embedding_lookup?$embedding_lookup/Read/ReadVariableOp?
$embedding_lookup/Read/ReadVariableOpReadVariableOp-embedding_lookup_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
??~
embedding_lookup/IdentityIdentity,embedding_lookup/Read/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
embedding_lookupResourceGather-embedding_lookup_read_readvariableop_resourceinputs%^embedding_lookup/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOp*
dtype0*+
_output_shapes
:?????????d?
embedding_lookup/Identity_1Identityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_lookup/Read/ReadVariableOp*+
_output_shapes
:?????????d?
embedding_lookup/Identity_2Identity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????d?
IdentityIdentity$embedding_lookup/Identity_2:output:0^embedding_lookup%^embedding_lookup/Read/ReadVariableOp*
T0*+
_output_shapes
:?????????d"
identityIdentity:output:0**
_input_shapes
:?????????d:2L
$embedding_lookup/Read/ReadVariableOp$embedding_lookup/Read/ReadVariableOp2$
embedding_lookupembedding_lookup: :& "
 
_user_specified_nameinputs
?2
?
C__inference_text_cnn_layer_call_and_return_conditional_losses_23466
input_1,
(embedding_statefulpartitionedcall_args_1)
%conv1d_statefulpartitionedcall_args_1)
%conv1d_statefulpartitionedcall_args_2+
'conv1d_1_statefulpartitionedcall_args_1+
'conv1d_1_statefulpartitionedcall_args_2+
'conv1d_2_statefulpartitionedcall_args_1+
'conv1d_2_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall? conv1d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?!embedding/StatefulPartitionedCall?
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1(embedding_statefulpartitionedcall_args_1*,
_gradient_op_typePartitionedCall-23403*M
fHRF
D__inference_embedding_layer_call_and_return_conditional_losses_23397*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:?????????d?
embedding/IdentityIdentity*embedding/StatefulPartitionedCall:output:0"^embedding/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????d?
conv1d/StatefulPartitionedCallStatefulPartitionedCallembedding/Identity:output:0%conv1d_statefulpartitionedcall_args_1%conv1d_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-23262*J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_23256*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*,
_output_shapes
:?????????a??
conv1d/IdentityIdentity'conv1d/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall*
T0*,
_output_shapes
:?????????a??
$global_max_pooling1d/PartitionedCallPartitionedCallconv1d/Identity:output:0*,
_gradient_op_typePartitionedCall-23342*X
fSRQ
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_23336*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:???????????
global_max_pooling1d/IdentityIdentity-global_max_pooling1d/PartitionedCall:output:0*
T0*(
_output_shapes
:???????????
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallembedding/Identity:output:0'conv1d_1_statefulpartitionedcall_args_1'conv1d_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-23292*L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_23286*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*,
_output_shapes
:?????????`??
conv1d_1/IdentityIdentity)conv1d_1/StatefulPartitionedCall:output:0!^conv1d_1/StatefulPartitionedCall*
T0*,
_output_shapes
:?????????`??
&global_max_pooling1d_1/PartitionedCallPartitionedCallconv1d_1/Identity:output:0*,
_gradient_op_typePartitionedCall-23360*Z
fURS
Q__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_23354*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:???????????
global_max_pooling1d_1/IdentityIdentity/global_max_pooling1d_1/PartitionedCall:output:0*
T0*(
_output_shapes
:???????????
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallembedding/Identity:output:0'conv1d_2_statefulpartitionedcall_args_1'conv1d_2_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-23322*L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_23316*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*,
_output_shapes
:?????????_??
conv1d_2/IdentityIdentity)conv1d_2/StatefulPartitionedCall:output:0!^conv1d_2/StatefulPartitionedCall*
T0*,
_output_shapes
:?????????_??
&global_max_pooling1d_2/PartitionedCallPartitionedCallconv1d_2/Identity:output:0*,
_gradient_op_typePartitionedCall-23378*Z
fURS
Q__inference_global_max_pooling1d_2_layer_call_and_return_conditional_losses_23372*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:???????????
global_max_pooling1d_2/IdentityIdentity/global_max_pooling1d_2/PartitionedCall:output:0*
T0*(
_output_shapes
:??????????Y
concatenate/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: ?
concatenate/concatConcatV2&global_max_pooling1d/Identity:output:0(global_max_pooling1d_1/Identity:output:0(global_max_pooling1d_2/Identity:output:0 concatenate/concat/axis:output:0*
T0*
N*(
_output_shapes
:??????????p
concatenate/IdentityIdentityconcatenate/concat:output:0*
T0*(
_output_shapes
:??????????n
dropout/IdentityIdentityconcatenate/Identity:output:0*
T0*(
_output_shapes
:??????????l
dropout/Identity_1Identitydropout/Identity:output:0*
T0*(
_output_shapes
:???????????
dense/StatefulPartitionedCallStatefulPartitionedCalldropout/Identity_1:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-23453*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_23447*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:??????????
dense/IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:??????????
IdentityIdentitydense/Identity:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*J
_input_shapes9
7:?????????d:::::::::2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall: : : : : :' #
!
_user_specified_name	input_1: : :	 : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????d<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:Ƽ
?
kernel_sizes
	embedding
	convs
max_poolings

classifier
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "TextCNN", "name": "text_cnn", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "TextCNN"}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
 "
trackable_list_wrapper
?

embeddings
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 100], "config": {"name": "embedding", "trainable": true, "batch_input_shape": [null, 100], "dtype": "float32", "input_dim": 30000, "output_dim": 20, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 100}}
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 384}}}}
?

beta_1

beta_2
	decay
 learning_rate
!itermpmqmr"ms#mt$mu%mv&mw'mxvyvzv{"v|#v}$v~%v&v?'v?"
	optimizer
_
0
"1
#2
$3
%4
&5
'6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
_
0
"1
#2
$3
%4
&5
'6
7
8"
trackable_list_wrapper
?
	variables
(non_trainable_variables
regularization_losses
)layer_regularization_losses

*layers
+metrics
	trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
3:1
??2text_cnn_1/embedding/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
	variables
,non_trainable_variables
regularization_losses
-layer_regularization_losses

.layers
/metrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

"kernel
#bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [4], "strides": [1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}}
?

$kernel
%bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [5], "strides": [1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}}
?

&kernel
'bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [6], "strides": [1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 20}}}}
?
<	variables
=regularization_losses
>trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GlobalMaxPooling1D", "name": "global_max_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "global_max_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GlobalMaxPooling1D", "name": "global_max_pooling1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "global_max_pooling1d_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GlobalMaxPooling1D", "name": "global_max_pooling1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "global_max_pooling1d_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
*:(	?2text_cnn_1/dense/kernel
#:!2text_cnn_1/dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
Hnon_trainable_variables
regularization_losses
Ilayer_regularization_losses

Jlayers
Kmetrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
/:-?2text_cnn_1/conv1d/kernel
%:#?2text_cnn_1/conv1d/bias
1:/?2text_cnn_1/conv1d_1/kernel
':%?2text_cnn_1/conv1d_1/bias
1:/?2text_cnn_1/conv1d_2/kernel
':%?2text_cnn_1/conv1d_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
'
L0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
0	variables
Mnon_trainable_variables
1regularization_losses
Nlayer_regularization_losses

Olayers
Pmetrics
2trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?
4	variables
Qnon_trainable_variables
5regularization_losses
Rlayer_regularization_losses

Slayers
Tmetrics
6trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?
8	variables
Unon_trainable_variables
9regularization_losses
Vlayer_regularization_losses

Wlayers
Xmetrics
:trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
<	variables
Ynon_trainable_variables
=regularization_losses
Zlayer_regularization_losses

[layers
\metrics
>trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
@	variables
]non_trainable_variables
Aregularization_losses
^layer_regularization_losses

_layers
`metrics
Btrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
D	variables
anon_trainable_variables
Eregularization_losses
blayer_regularization_losses

clayers
dmetrics
Ftrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	etotal
	fcount
g
_fn_kwargs
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
h	variables
lnon_trainable_variables
iregularization_losses
mlayer_regularization_losses

nlayers
ometrics
jtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
8:6
??2&Adam/text_cnn_1/embedding/embeddings/m
/:-	?2Adam/text_cnn_1/dense/kernel/m
(:&2Adam/text_cnn_1/dense/bias/m
4:2?2Adam/text_cnn_1/conv1d/kernel/m
*:(?2Adam/text_cnn_1/conv1d/bias/m
6:4?2!Adam/text_cnn_1/conv1d_1/kernel/m
,:*?2Adam/text_cnn_1/conv1d_1/bias/m
6:4?2!Adam/text_cnn_1/conv1d_2/kernel/m
,:*?2Adam/text_cnn_1/conv1d_2/bias/m
8:6
??2&Adam/text_cnn_1/embedding/embeddings/v
/:-	?2Adam/text_cnn_1/dense/kernel/v
(:&2Adam/text_cnn_1/dense/bias/v
4:2?2Adam/text_cnn_1/conv1d/kernel/v
*:(?2Adam/text_cnn_1/conv1d/bias/v
6:4?2!Adam/text_cnn_1/conv1d_1/kernel/v
,:*?2Adam/text_cnn_1/conv1d_1/bias/v
6:4?2!Adam/text_cnn_1/conv1d_2/kernel/v
,:*?2Adam/text_cnn_1/conv1d_2/bias/v
?2?
 __inference__wrapped_model_23237?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????d
?2?
(__inference_text_cnn_layer_call_fn_23484?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????d
?2?
C__inference_text_cnn_layer_call_and_return_conditional_losses_23466?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????d
?2?
)__inference_embedding_layer_call_fn_23524?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_embedding_layer_call_and_return_conditional_losses_23518?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_23542?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_23535?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
2B0
#__inference_signature_wrapper_23506input_1
?2?
&__inference_conv1d_layer_call_fn_23267?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?2?
A__inference_conv1d_layer_call_and_return_conditional_losses_23256?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?2?
(__inference_conv1d_1_layer_call_fn_23297?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?2?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_23286?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?2?
(__inference_conv1d_2_layer_call_fn_23327?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?2?
C__inference_conv1d_2_layer_call_and_return_conditional_losses_23316?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"??????????????????
?2?
4__inference_global_max_pooling1d_layer_call_fn_23345?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_23336?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
6__inference_global_max_pooling1d_1_layer_call_fn_23363?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
Q__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_23354?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
6__inference_global_max_pooling1d_2_layer_call_fn_23381?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
Q__inference_global_max_pooling1d_2_layer_call_and_return_conditional_losses_23372?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
 __inference__wrapped_model_23237r	"#$%&'0?-
&?#
!?
input_1?????????d
? "3?0
.
output_1"?
output_1??????????
(__inference_text_cnn_layer_call_fn_23484W	"#$%&'0?-
&?#
!?
input_1?????????d
? "???????????
C__inference_text_cnn_layer_call_and_return_conditional_losses_23466d	"#$%&'0?-
&?#
!?
input_1?????????d
? "%?"
?
0?????????
? 
)__inference_embedding_layer_call_fn_23524R/?,
%?"
 ?
inputs?????????d
? "??????????d?
D__inference_embedding_layer_call_and_return_conditional_losses_23518_/?,
%?"
 ?
inputs?????????d
? ")?&
?
0?????????d
? y
%__inference_dense_layer_call_fn_23542P0?-
&?#
!?
inputs??????????
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_23535]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
#__inference_signature_wrapper_23506}	"#$%&';?8
? 
1?.
,
input_1!?
input_1?????????d"3?0
.
output_1"?
output_1??????????
&__inference_conv1d_layer_call_fn_23267j"#<?9
2?/
-?*
inputs??????????????????
? "&?#????????????????????
A__inference_conv1d_layer_call_and_return_conditional_losses_23256w"#<?9
2?/
-?*
inputs??????????????????
? "3?0
)?&
0???????????????????
? ?
(__inference_conv1d_1_layer_call_fn_23297j$%<?9
2?/
-?*
inputs??????????????????
? "&?#????????????????????
C__inference_conv1d_1_layer_call_and_return_conditional_losses_23286w$%<?9
2?/
-?*
inputs??????????????????
? "3?0
)?&
0???????????????????
? ?
(__inference_conv1d_2_layer_call_fn_23327j&'<?9
2?/
-?*
inputs??????????????????
? "&?#????????????????????
C__inference_conv1d_2_layer_call_and_return_conditional_losses_23316w&'<?9
2?/
-?*
inputs??????????????????
? "3?0
)?&
0???????????????????
? ?
4__inference_global_max_pooling1d_layer_call_fn_23345jE?B
;?8
6?3
inputs'???????????????????????????
? "!????????????????????
O__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_23336wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+
$?!
0??????????????????
? ?
6__inference_global_max_pooling1d_1_layer_call_fn_23363jE?B
;?8
6?3
inputs'???????????????????????????
? "!????????????????????
Q__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_23354wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+
$?!
0??????????????????
? ?
6__inference_global_max_pooling1d_2_layer_call_fn_23381jE?B
;?8
6?3
inputs'???????????????????????????
? "!????????????????????
Q__inference_global_max_pooling1d_2_layer_call_and_return_conditional_losses_23372wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+
$?!
0??????????????????
? 