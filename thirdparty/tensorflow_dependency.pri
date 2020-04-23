isEmpty(TENSORFLOW_DIR):TENSORFLOW_DIR=$$(TENSORFLOW_DIR)


isEmpty(TENSORFLOW_DIR) {
#    message("set TENSORFLOW_DIR as environment variable or qmake variable to get rid of this message")
    TENSORFLOW_DIR=/home/pgcs-04/install/tensorflow/tensorflow-1.14.0
#    TENSORFLOW_DIR=/home/dat/setupfiles/tensorflow1.4/tensorflow-1.14.0
}

!exists($$TENSORFLOW_DIR):error("No tensorflow dir found - set TENSORFLOW_DIR to enable.")
    
INCLUDEPATH += $$TENSORFLOW_DIR
INCLUDEPATH += $$TENSORFLOW_DIR/tensorflow
INCLUDEPATH += $$TENSORFLOW_DIR/bazel-tensorflow-1.14.0/external/eigen_archive
INCLUDEPATH += $$TENSORFLOW_DIR/bazel-tensorflow-1.14.0/external/protobuf_archive/src
INCLUDEPATH += $$TENSORFLOW_DIR/bazel-genfiles

LIBS += -L$$TENSORFLOW_DIR/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework
