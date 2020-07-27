#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

int main() {
	// Load names of classes
	vector<string> classNamesVec;
	string classesFile = "H:/YOLOv4_Mask/voc-mask.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classNamesVec.push_back(line);

	// Give the configuration and weight files for the model
	String yolov4_config = "H:/YOLOv4_Mask/yolov4-tiny-mask.cfg";
	String yolov4_model = "H:/YOLOv4_Mask/yolov4-tiny-mask_best.weights";

	//String yolov4_config = "H:/YOLOv4_Mask/yolov4-mask.cfg";
	//String yolov4_model = "H:/YOLOv4_Mask/yolov4-mask_best.weights";

	Net net = readNetFromDarknet(yolov4_config, yolov4_model);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
	std::vector<String> outNames = net.getUnconnectedOutLayersNames();
	for (int i = 0; i < outNames.size(); i++) {
	    printf("output layer name : %s\n", outNames[i].c_str());
	}
	
	//VideoCapture capture(0);
	//capture.open("H:/YOLOv4_Mask/kouzhao1.mp4");
	//Mat frame;
	// ¼ÓÔØÍ¼Ïñ 
	while (true) {
		int64 start = getTickCount();
		//capture.read(frame);
		Mat frame = imread("H:/YOLOv4_Mask/imgs/img7.jpg");
		Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false);
		net.setInput(inputBlob);
	
		// ¼ì²â
		std::vector<Mat> outs;
		net.forward(outs, outNames);

		vector<Rect> boxes;
		vector<int> classIds;
		vector<float> confidences;
		for (size_t i = 0; i<outs.size(); ++i){
			//detected objects and C is a number of classes + 4 where the first 4
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols){
				Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				Point classIdPoint;
				double confidence;
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > 0.5){
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;
					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(Rect(left, top, width, height));
				}
			}
		}
		vector<int> indices;
		NMSBoxes(boxes, confidences, 0.5, 0.2, indices);
		for (size_t i = 0; i < indices.size(); ++i){
			int idx = indices[i];
			Rect box = boxes[idx];
			String className = classNamesVec[classIds[idx]];
			putText(frame, className.c_str(), box.tl(), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2, 8);
			rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
		}
		float fps = getTickFrequency() / (getTickCount() - start);
		float time = (getTickCount() - start) / getTickFrequency();
		ostringstream ss;
		ss << "FPS : " << fps << " detection time: " << time * 1000 << " ms";
		putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));
		imshow("YOLOv4-Detections", frame);
		char c = waitKey(1);
		if (c == 27) {
			break;
		}
	}
	waitKey(0);
	return 0;
}