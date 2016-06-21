/*
*
*
Dependencies:
1. Training Image Folder with Birds (------- Place these in the folder images/birds/ with naming convention : frame0001.jpg -------)
2. Training Image Folder with Squirrels (------- Place these in the folder images/squirrel/ with naming convention : frame0001.jpg -------)

****** We have written a python script which will rename the files in a numeric order. Please use it to rename the files. ******

Command Line Arguments:
1. Test Video File (------- path/filename.avi -------)
*
*
*/

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

int gridX = 0, gridY = 0, gridX2 = 0, gridY2 = 0;
int drawing = false;

vector<Mat> createTrainingData();

static void onMouse(int event, int x, int y, int, void*) {
	if (event == EVENT_MOUSEMOVE){
		if (drawing){
			gridX2 = x;
			gridY2 = y;
		}
	}
	else if (event == EVENT_LBUTTONDOWN) {
		drawing = true;
		gridX = x;
		gridY = y;
	}
	else {
		drawing = false;
	}
}
int main(int argc, char *argv[])
{
	if(argc < 2)
		cout << "Invalid number of arguments" << endl;
	
	vector<Mat> trainingData = createTrainingData();

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::POLY;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	params.degree = 2;
	params.gamma = 3;

	CvSVM SVM;
	SVM.train(trainingData[1], trainingData[0], Mat(), Mat(), params);
	
	VideoCapture vid(argv[1]);
	if (!vid.isOpened())
		return -1;

	Mat previousFrame, currentFrame, nextFrame, diff1, diff2, motion, result, resultFloat;
	int frameCount = 0;

	vid >> previousFrame;
	if(previousFrame.empty())
		cout << "Invalid Filename" << endl;
	gridX = 0, gridY = 0, gridX2 = 0, gridY2 = 0;
	drawing = false;
	String window = "Please Select Region of Interest using the mouse (Draw a Rectangle around region) and then press <space>";
	namedWindow(window, WINDOW_AUTOSIZE);

	setMouseCallback(window, onMouse, 0);
	imshow(window, previousFrame);
	waitKey(0);
	Rect regionOfInterest(Point(gridX, gridY), Point(gridX2, gridY2));
	previousFrame = previousFrame(regionOfInterest);
	imshow(window, previousFrame);
	waitKey(0);
	destroyAllWindows();

	
	cvtColor(previousFrame, previousFrame, CV_RGB2GRAY);

	vid >> currentFrame;
	currentFrame = currentFrame(regionOfInterest);
	cvtColor(currentFrame, currentFrame, CV_RGB2GRAY);

	namedWindow("Motion", WINDOW_AUTOSIZE);

	vector<Mat> trainingImages;
	int name = 0;
	while (true)
	{
		vid >> nextFrame;
		if (nextFrame.empty())
			break;
		nextFrame = nextFrame(regionOfInterest);
		result = nextFrame;
		nextFrame.convertTo(resultFloat, CV_32FC3);
		cvtColor(nextFrame, nextFrame, CV_RGB2GRAY);

		absdiff(previousFrame, currentFrame, diff1);
		absdiff(previousFrame, nextFrame, diff2);
		bitwise_and(diff1, diff2, motion);
		threshold(motion, motion, 150, 255, CV_THRESH_BINARY);
		erode(motion, motion, Mat());
		dilate(motion, motion, Mat());
		float motionArea = 0.0, motionBlue = 0.0, motionGreen = 0.0, motionRed = 0.0;
		int minX = motion.cols, maxX = 0, minY = motion.rows, maxY = 0;
		for (int i = 0; i < motion.rows; i++){
			for (int j = 0; j < motion.cols; j++){
				if (motion.at<uchar>(i, j) == 255)
				{
					motionArea++;
					Vec3f vals = resultFloat.at<Vec3f>(i, j);
					motionBlue += vals.val[0];
					motionGreen += vals.val[1];
					motionRed += vals.val[2];
					if (minX > i) minX = i;
					if (maxX < i) maxX = i;
					if (minY > j) minY = j;
					if (maxY < j) maxY = j;
				}
			}
		}
		if (motionArea > 50) {
			Mat test = (Mat_<float>(1, 4) << motionArea, (float) motionBlue / motionArea, (float) motionGreen / motionArea, (float) motionRed / motionArea);
			if ((int) SVM.predict(test) == 1){
				if (minX - 10 > 0) minX -= 10;
				if (minY - 10 > 0) minY -= 10;
				if (maxX + 10 < result.cols - 1) maxX += 10;
				if (maxY + 10 < result.rows - 1) maxY += 10;
				Point x(minX, minY);
				Point y(maxX, maxY);
				Rect rect(x, y);
				rectangle(result, rect, Scalar(0, 0, 255), 1);
			} else if((int) SVM.predict(test) == 0){
				if (minX - 10 > 0) minX -= 10;
				if (minY - 10 > 0) minY -= 10;
				if (maxX + 10 < result.cols - 1) maxX += 10;
				if (maxY + 10 < result.rows - 1) maxY += 10;
				Point x(minX, minY);
				Point y(maxX, maxY);
				Rect rect(x, y);
				rectangle(result, rect, Scalar(255, 0, 0), 1);
			}
		}
		//imshow("Motion", result);
		imwrite("images/results/"+to_string(name)+".jpg",result);
		if (waitKey(30) >= 0) break;
		previousFrame = currentFrame;
		currentFrame = nextFrame;
		
		name++;
	}
	return 0;
}

vector<Mat> createTrainingData() {
	VideoCapture birds("images/birds/frame%04d.jpg");
	VideoCapture squirrel("images/squirrel/frame%04d.jpg");

	Mat previousFrame, currentFrame, nextFrame, diff1, diff2, motion, result;

	birds >> previousFrame;
	
	String window = "Please Select Region of Interest using the mouse and press space. (Draw a Rectangle around region)";
	namedWindow(window, WINDOW_AUTOSIZE);
	
	setMouseCallback(window, onMouse, 0);

	imshow(window, previousFrame);
	waitKey(0);

	Rect regionOfInterest(Point(gridX, gridY), Point(gridX2, gridY2));
	previousFrame = previousFrame(regionOfInterest);

	imshow(window, previousFrame);
	waitKey(0);

	destroyAllWindows();

	cvtColor(previousFrame, previousFrame, CV_RGB2GRAY);

	birds >> currentFrame;
	currentFrame = currentFrame(regionOfInterest);

	cvtColor(currentFrame, currentFrame, CV_RGB2GRAY);

	vector<Mat> trainingImages;
	vector<vector<float>> features;

	while (true)
	{
		birds >> nextFrame;

		if (nextFrame.empty())
			break;

		nextFrame = nextFrame(regionOfInterest);
		nextFrame.convertTo(result, CV_32FC3);

		cvtColor(nextFrame, nextFrame, CV_RGB2GRAY);

		absdiff(previousFrame, currentFrame, diff1);
		absdiff(previousFrame, nextFrame, diff2);

		bitwise_and(diff1, diff2, motion);

		threshold(motion, motion, 150, 255, CV_THRESH_BINARY);

		erode(motion, motion, Mat());
		dilate(motion, motion, Mat());

		float motionArea = 0.0, motionBlue = 0.0, motionGreen = 0.0, motionRed = 0.0;
		for (int i = 0; i < motion.rows; i++){
			for (int j = 0; j < motion.cols; j++){
				if (motion.at<uchar>(i, j) == 255)
				{
					motionArea++;
					Vec3f vals = result.at<Vec3f>(i, j);
					motionBlue += vals.val[0];
					motionGreen += vals.val[1];
					motionRed += vals.val[2];
				}
			}
		}
		if (motionArea > 10) {
			vector<float> feature{ motionArea, (float) motionBlue / motionArea, (float) motionGreen / motionArea, (float) motionRed / motionArea };
			features.push_back(feature);
		}
		previousFrame = currentFrame;
		currentFrame = nextFrame;
	}

	int birdCount = (int)features.size();
	
	squirrel >> previousFrame;
	previousFrame = previousFrame(regionOfInterest);
	cvtColor(previousFrame, previousFrame, CV_RGB2GRAY);

	squirrel >> currentFrame;
	currentFrame = currentFrame(regionOfInterest);
	cvtColor(currentFrame, currentFrame, CV_RGB2GRAY);

	while (true)
	{
		squirrel >> nextFrame;

		if (nextFrame.empty())
			break;

		nextFrame = nextFrame(regionOfInterest);
		nextFrame.convertTo(result, CV_32FC3);

		cvtColor(nextFrame, nextFrame, CV_RGB2GRAY);

		absdiff(previousFrame, currentFrame, diff1);
		absdiff(previousFrame, nextFrame, diff2);

		bitwise_and(diff1, diff2, motion);

		threshold(motion, motion, 150, 255, CV_THRESH_BINARY);

		erode(motion, motion, Mat());
		dilate(motion, motion, Mat());

		float motionArea = 0.0, motionBlue = 0.0, motionGreen = 0.0, motionRed = 0.0;
		for (int i = 0; i < motion.rows; i++){
			for (int j = 0; j < motion.cols; j++){
				if (motion.at<uchar>(i, j) == 255)
				{
					motionArea++;
					Vec3f vals = result.at<Vec3f>(i, j);
					motionBlue += vals.val[0];
					motionGreen += vals.val[1];
					motionRed += vals.val[2];
				}
			}
		}
		if (motionArea > 10) {
			vector<float> feature{ motionArea, (float) motionBlue / motionArea, (float) motionGreen / motionArea, (float) motionRed / motionArea };
			features.push_back(feature);
		}
		previousFrame = currentFrame;
		currentFrame = nextFrame;
	}

	vector<Mat> train;

	Mat labels(features.size(), 1, CV_32FC1, 1);
	for (int i = 0; i < birdCount; i++) {
		labels.at<float>(i, 0) = 0;
	}

	Mat trainingData(features.size(), 4, CV_32FC1);
	for (int i = 0; i < features.size(); i++) {
		for (int j = 0; j < 4; j++) {
			trainingData.at<float>(i, j) = features[i][j];
		}
	}

	train.push_back(labels);
	train.push_back(trainingData);

	return train;
}
