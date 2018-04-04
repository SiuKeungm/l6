#include <opencv2/opencv.hpp>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

// this program shows how to use optical flow

string file_1 = "./1.png";  // first image
string file_2 = "./2.png";  // second image

// TODO implement this funciton
/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse = false
);

// TODO implement this funciton
/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse = false
);

/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return
 */
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[(int)img.step + 1]
    );
}


int main(int argc, char **argv) {

    // images, note they are CV_8UC1, not CV_8UC3
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    // key points, using GFTT here.
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(img1, kp1);

    // now lets track these key points in the second image
    // first use single level LK in the validation picture
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single,true);

    // then test multi-level LK
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi,true);

    // use opencv's flow for validation
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error, cv::Size(8, 8));

    // plot the differences of those functions
    Mat img2_single;
    cv::cvtColor(img2, img2_single, CV_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, CV_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, CV_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    /*cv::imwrite("inverse_single_level.png",img2_single);
    cv::imwrite("inverse_multi_level.png",img2_multi);*/
    cv::waitKey(0);

    return 0;
}

void OpticalFlowSingleLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse
) {

    // parameters
    int half_patch_size = 4;
    int iterations = 10;
    bool have_initial = !kp2.empty();

    for (size_t i = 0; i < kp1.size(); i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (have_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; iter++) {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (kp.pt.x + dx <= half_patch_size || kp.pt.x + dx >= img1.cols - half_patch_size ||
                kp.pt.y + dy <= half_patch_size || kp.pt.y + dy >= img1.rows - half_patch_size) {   // go outside
                succ = false;
                break;
            }

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {

                    // TODO START YOUR CODE HERE (~8 lines)

                    double error=0;
                    //!!!!!!!!!!注意误差函数的定义是窗口中所有像素点的误差
                    error = GetPixelValue(img1,kp.pt.x+x,kp.pt.y+y)-GetPixelValue(img2,kp.pt.x+dx+x,kp.pt.y+dy+y);
                    Eigen::Matrix<double ,1,2> J;  // Jacobian
                    if (inverse == false) {
                        J(0,0)=-(GetPixelValue(img2,kp.pt.x+dx+x+1,kp.pt.y+dy+y)-GetPixelValue(img2,kp.pt.x+dx+x-1,kp.pt.y+dy+y))/2;
                        J(0,1)=-(GetPixelValue(img2,kp.pt.x+dx+x,kp.pt.y+dy+y+1)-GetPixelValue(img2,kp.pt.x+dx+x,kp.pt.y+dy+y-1))/2;
                        // Forward Jacobian
                    } else {
                        // Inverse Jacobian
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J(0,0) = -(GetPixelValue(img1,kp.pt.x+x+1,kp.pt.y+y)-GetPixelValue(img1,kp.pt.x+x-1,kp.pt.y+y))/2;
                        J(0,1) = -(GetPixelValue(img1,kp.pt.x+x,kp.pt.y+y+1)-GetPixelValue(img1,kp.pt.x+x,kp.pt.y+y-1))/2;
                    }
                    // compute H, b and set cost;
                    H +=J.transpose() * J;
                    b += -J.transpose() * error;
                    cost +=error*error;
                    // TODO END YOUR CODE HERE
                }

            // compute update
            // TODO START YOUR CODE HERE (~1 lines)
            Eigen::Vector2d update;

            update = H.ldlt().solve(b);
            // TODO END YOUR CODE HERE

            if (std::isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost) {
                cout << "cost increased: " << cost << ", " << lastCost << endl;
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;
        }

        success.push_back(succ);

        // set kp2
        if (have_initial) {
            kp2[i].pt = kp.pt + Point2f(dx, dy);
        } else {
            KeyPoint tracked = kp;
            tracked.pt += cv::Point2f(dx, dy);
            kp2.push_back(tracked);
        }
    }
}

void OpticalFlowMultiLevel(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &kp1,
        vector<KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse) {

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<Mat> pyr1(4), pyr2(4); // image pyramids
    // TODO START YOUR CODE HERE (~8 lines)
    for (int i = 0; i < pyramids; i++) {
        Size sz(cvRound(img1.cols*scales[i]),cvRound(img1.rows*scales[i]));
        Mat temp(sz,img1.type());

        temp.copyTo(pyr1[i]);//这里要进行深拷贝，不然pyr1和pyr2共享一块内存
        temp.copyTo(pyr2[i]);
        if(i != 0){
            resize(pyr1[i-1],pyr1[i],sz);
            resize(pyr2[i-1],pyr2[i],sz);

        } else{
            img1.copyTo(pyr1[i]);
            img2.copyTo(pyr2[i]);
        }

    }
    // TODO END YOUR CODE HERE
    //示范代码
    // TODO START YOUR CODE HERE (~8 lines)
    /*for (int i = 0; i < pyramids; i++) {
        Mat img1_tmp, img2_tmp;
        resize(img1,img1_tmp,Size(img1.cols * scales[i],img1.rows * scales[i]));
        resize(img2,img2_tmp,Size(img2.cols * scales[i],img2.rows * scales[i]));
        pyr1.push_back(img1_tmp);
        pyr2.push_back(img2_tmp);
    }*/
    // TODO END YOUR CODE HERE


    // coarse-to-fine LK tracking in pyramids
    // TODO START YOUR CODE HERE
    vector<vector<KeyPoint>> kp1_ths(4),kp2_ths(4);
    vector<vector<bool>> success_ths(4);
    for(int i = pyramids-1; i >=0; i--){

        for(auto kp:kp1) {
            KeyPoint kptemp;
            kptemp.pt = kp.pt * scales[i];
            kp1_ths[i].push_back(kptemp);
        }
        if(i<pyramids-1) {
            for (auto kp:kp2_ths[i + 1]) {
                KeyPoint kptemp = kp;
                kptemp.pt /= pyramid_scale;
                kp2_ths[i].push_back(kptemp);
            }

        }

            OpticalFlowSingleLevel(pyr1[i],pyr2[i],kp1_ths[i],kp2_ths[i],success_ths[i],inverse);
    }

    kp2 = kp2_ths[0];
    success = success_ths[0];
    // TODO END YOUR CODE HERE
    //示范代码
    // TODO START YOUR CODE HERE

    /*vector<KeyPoint> kp2_now,kp2_last;
    vector<bool> succ;
    for(int i=pyramids-1;i>=0;i--){
        vector<KeyPoint> vkp1;
        for(int j=0;j < kp1.size();j++){
            KeyPoint kp1_tmp = kp1[j];
            kp1_tmp.pt *=scales[i];
            vkp1.push_back(kp1_tmp);
            if(i < pyramids-1){
                KeyPoint kp2_tmp = kp2_last[j];
                kp2_tmp.pt /=pyramid_scale;
                kp2_now.push_back(kp2_tmp);
            }
        }
        kp2_last.clear();
        succ.clear();
        OpticalFlowSingleLevel(pyr1[i],pyr2[i],vkp1,kp2_now,succ,inverse);
        kp2_last.swap(kp2_now);
    }
    kp2 = kp2_last;
    success=succ;*/
    // TODO END YOUR CODE HERE
    // don't forget to set the results into kp2
}
