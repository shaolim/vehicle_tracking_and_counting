#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

using namespace cv;
using namespace std;

/// Function header
void foregroundAnalysis(Mat&, Mat&, Mat&);
void mask(Mat&, Mat&);

  /** @function main */
int main( int argc, char** argv ){
    Mat frame, objDetect;
    const int thresh = 35;
    Mat element( 5, 5, CV_8U, Scalar(1) );
    bool debugMode = false;
    bool pause = false;

    // open background image
    Mat background = imread("input/background.png");

    // open the video file for reading
    VideoCapture cap("input/highwayII_raw.avi");

    if(!cap.isOpened()){
        cout << "Cannot open the video file" << endl;
        return -1;
    }

    while(1){
        Mat difference, gray, threshold_output, frame_mask, background_mask;

        // read a new frame from video
        bool bSuccess = cap.read(frame);

        // if no success, break loop
        if(!bSuccess){
          cout << "Cannot read the frame from video file" << '\n';
          break;
        }

        imshow( "Source Window", frame );

        // masking
        mask(background, background_mask);
        mask(frame, frame_mask);

        line(frame, Point(30, 90), Point(240, 90), Scalar(0, 0, 255), 2, 8, 0);

        // background substraction
        absdiff(frame_mask, background_mask, difference);
        cvtColor(difference, gray, CV_BGR2GRAY);

        GaussianBlur(gray, gray, Size(5,5), 1.5);

        threshold( gray, threshold_output, thresh, 255, THRESH_BINARY );

        morphologyEx( threshold_output, threshold_output, MORPH_CLOSE, element );

        if(debugMode){
            imshow("Background Substraction", threshold_output);
            imshow("Gray Scale", gray);
        } else {
            destroyWindow("Background Substraction");
            destroyWindow("Gray Scale");
        }

        /*std::vector< cv::Point2f > points[2];
        TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
        Size s(10,10), w(31,31);
        int maxCorners = 10;
        double qualityLevel = 0.01;
        double minDistance = 20.;
        cv::Mat mask;
        int blockSize = 3;
        bool useHarrisDetector = false;
        double k = 0.04;

        goodFeaturesToTrack( threshold_output, points[1], maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k );
        //Mat corners = points[1].getMat();
        //int ncorners = corners.checkVector(2);
        //cout << ncorners.checkVector(2) << " ";
        //cornerSubPix(threshold_output, points[1], s, Size(-1,-1), termcrit);

        //threshold_output_prev = threshold_output.clone();

        vector<uchar> status;
        vector<float> err;

        //calcOpticalFlowPyrLK(threshold_output_prev, threshold_output, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);

        for( size_t i = 0; i < points[1].size(); i++ ){
            cv::circle( frame, points[1][i], 3, cv::Scalar(255, 0, 0), -1 );
        }*/

        foregroundAnalysis(frame, threshold_output, objDetect);
        imshow("Foreground", objDetect);

        char c = (char)waitKey(30);
        if( c == 27 )
            break;
        switch(c){
      		case 'd': //'d' has been pressed. this will debug mode
      			debugMode = !debugMode;
      			if(debugMode == false) cout<<"Debug mode disabled."<<endl;
      			else cout<<"Debug mode enabled."<<endl;
      			break;
      		/*case 'p': //'p' has been pressed. this will pause/resume the code.
                pause = !pause;
                if(pause == true){ cout<<"Code paused, press 'p' again to resume"<<endl;
                    while (pause == true){
                        //stay in this loop until
                        switch (waitKey()){
                          //a switch statement inside a switch statement? Mind blown.
                          case 112:
                          //change pause back to false
                            pause = false;
                            cout<<"Code Resumed"<<endl;
                            break;
                        }
                    }   
                } */      	
        }

    }

    return(0);
}

/** @function foregroundAnalysis*/
void foregroundAnalysis(Mat& frame, Mat& threshold_output, Mat& objDetect){
    static int count=0;
    double area, ar;
    Mat output, vehicle_ROI;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    frame.copyTo(output);

    /// Find contours
    findContours( threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );

    for( int i = 0; i < contours.size(); i++ ) {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 5, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
    }

    /// Draw polygonal contour + bonding rects + circles
    Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ ){
        Scalar color = Scalar( 255, 0, 0);
        drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        rectangle( output, boundRect[i].tl(), boundRect[i].br(), color, 1, 8, 0 );
    
        // get the center point of blob
        Point center((0.5 * (boundRect[i].tl().x+boundRect[i].br().x)), 0.5 * (boundRect[i].tl().y+boundRect[i].br().y)); 
        circle( output, center, 3, cv::Scalar(255, 0, 0), -1 );
        if (center.y > 86 && center.y < 94){//if(area > 450.0 && ar > 0.8) { 
            count++;
            cout << count << endl;
        }
    }

    // print text to window
    stringstream ss;
        ss << count;
        string s = ss.str();
        int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontScale = 1;
        int thickness = 2;
        cv::Point textOrg(10, 130);
        cv::putText(output, s, textOrg, fontFace, fontScale, Scalar(0,255,0), thickness,5);

    output.copyTo(objDetect);

    /// Show in a window
    imshow("Contours", drawing );
}

// find region of interest (ROI)
void mask(Mat& src, Mat& dest ){
    //Mat black(src.rows, src.cols, src.type(), cv::Scalar::all(0));
    Mat mask(src.rows, src.cols, CV_8UC1, cv::Scalar(0));

    vector< vector<Point> >  co_ordinates;
    co_ordinates.push_back(vector<Point>());
    co_ordinates[0].push_back(Point(91, 52));
    co_ordinates[0].push_back(Point(228, 48));
    co_ordinates[0].push_back(Point(210, 239));
    co_ordinates[0].push_back(Point(1, 239));
    co_ordinates[0].push_back(Point(1, 126));
   
    drawContours( mask,co_ordinates,0, Scalar(255),CV_FILLED, 8 ); //

    src.copyTo(dest);
    for (int y = 0; y < src.rows; ++y){
        for (int x = 0; x < src.cols; ++x){
            if (mask.at<uchar>(y,x) == 0)
            {
                for (int i = 0; i < 3; ++i)
                    dest.at<Vec3b>(y,x)[i] = 0;
            }
        }
    }
}