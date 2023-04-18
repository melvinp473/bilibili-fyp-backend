import weka.core.jvm as jvm
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.classifiers import PredictionOutput, KernelClassifier, Kernel, Evaluation

jvm.start(system_cp=True, packages=True, max_heap_size="512m")

ml_data_dir = "C:/Users/Asus/Documents/FIT4701/FYP_repo/bilibili-fyp-backend/src/WEKA/datasets/"
loader = Loader(classname="weka.core.converters.ArffLoader")
ml_data = loader.load_file(ml_data_dir + "MLDATA.csv.arff")
ml_data.class_is_last()
print(ml_data)

cls = KernelClassifier(classname="weka.classifiers.functions.SMOreg", options=["-N", "0"])
kernel = Kernel(classname="weka.classifiers.functions.supportVector.RBFKernel", options=["-G", "0.1"])
cls.kernel = kernel
pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
evl = Evaluation(ml_data)
evl.crossvalidate_model(cls, ml_data, 10, Random(1), pout)

print(evl.summary())
print(pout.buffer_content())

jvm.stop()