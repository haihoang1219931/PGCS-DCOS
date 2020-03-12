#include "preprocessing.h"

//===============================
// I. Main Functions            "
//===============================
std::vector<cv::Mat> preprocess(cv::Mat grayImg, int plateType, int *sign)
{
    cv::Mat binImg;
    cv::Mat filImg;
    if((grayImg.rows > 36 && plateType == 0) || (grayImg.rows > 54 && plateType == 1))
    {
        binarize(grayImg, binImg, 7);
        grayImg.release();
        filter(binImg, filImg);
        binImg.release();
    }
    else
    {
        binarize(grayImg, binImg, 3);
        grayImg.release();
        filImg = binImg.clone();
        binImg.release();
    }
//    cv::imwrite("img/plateImage" + std::to_string(rand()) + ".png", filImg, {CV_IMWRITE_PNG_COMPRESSION, 0});

    cv::Mat prunedImg;
    std::vector<cv::Mat> chars;
    cv::Mat temp;
    horiPrune(filImg, temp, 0.75);
    if(temp.empty())
    {
        return chars;
    }
    if(plateType == 0)
    {
        vertPrune(temp, prunedImg, plateType);
        chars = extractCharsLP(prunedImg, sign);
    }
    else if(plateType == 1)
    {
//        printf("%s==================================plateType == 1\n", __func__);
        vertPrune(temp, prunedImg, plateType);
        chars = extractCharsSP(prunedImg, sign);
    }
//    cv::imwrite("img/plateImage" + std::to_string(rand()) + ".png", prunedImg, {CV_IMWRITE_PNG_COMPRESSION, 0});
    filImg.release();
    prunedImg.release();
//    printf("%d\n", chars.size());
    return chars;
}

std::vector<cv::Mat> extractCharsLP(cv::Mat input, int *sign)
{
    *sign = -1;
    std::vector<cv::Mat> chars;
    int width = input.cols;
    int height = input.rows;

    if(height < 16)
        return chars;

    // == End pruning ./.
    // 2. Get popular rectangles
    cv::Mat thresMat = ~input;
    std::vector<std::vector<cv::Point>> contoursPoints;
    std::vector<cv::Vec4i> hierrachy;
    cv::findContours(thresMat, contoursPoints, hierrachy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    thresMat.release();
    int numContoursPoints = contoursPoints.size();
    if (numContoursPoints > 0) {
        std::vector<cv::Rect> rects;
        for (int i = 0; i < numContoursPoints; i++) {
            cv::Rect rect = cv::boundingRect(contoursPoints[i]);

            if (rect.height > 12 && rect.width > 3 && rect.width < 0.5 * width && rect.height > 0.5 * height) {
                rects.push_back(rect);
            }
        }
        std::vector<std::pair<float, int>> stats;
        int numRect = rects.size();
        // Group rectangles that have the same height
        if(numRect < 2)
            return chars;
        stats.push_back(std::make_pair((float)rects[0].height, 1));
        for (int i = 1; i < numRect; i++) {
            int numStats = stats.size();
            bool inserted = false;
            for (int j = 0; j < numStats; j++) {
                if (std::abs(stats[j].first - rects[i].height)  < 4) {
                    stats[j].first = (stats[j].first * stats[j].second + (float)rects[i].height) / (stats[j].second + 1);
                    stats[j].second++;
                    inserted = true;
                    break;
                }
            }

            if (!inserted)
                stats.push_back(std::make_pair((float)rects[i].height, 1));
        }
        int numStats = stats.size();
        int maxCount = stats[0].second;
        int stdHeight = stats[0].first;

        for (int i = 1; i < numStats; i++) {
            if (stats[i].second > maxCount) {
                maxCount = stats[i].second;
                stdHeight = stats[i].first;
            }
        }

        std::vector<cv::Rect> stdRects;

        for (int i = 0; i < numRect; i++) {
            if (std::abs(stdHeight - (float)rects[i].height) < 2.5) {
                stdRects.push_back(rects[i]);
            }
        }
        rects.clear();

        // ==> End 2 ./.
        // 3. Remove inner rectangles and extra-get predicted rectangles
        std::sort(stdRects.begin(), stdRects.end(), [](cv::Rect a, cv::Rect b) {
            return a.x < b.x;
        });
        std::vector<cv::Rect> newRects;
        newRects.push_back(stdRects[0]);

        for (uint i = 1; i < stdRects.size(); i++) {
            if (stdRects[i].x > stdRects[i - 1].x + 0.75 * stdRects[i - 1].width && stdRects[i].x + stdRects[i].width > stdRects[i - 1].x + stdRects[i - 1].width) {
                newRects.push_back(stdRects[i]);
            }
        }

        stdRects.clear();
        int count = 0;
        float sumWidth = 0;

        for (uint i = 0; i < newRects.size(); i++) {
            if (newRects[i].width < newRects[i].height) {
                sumWidth += newRects[i].width;
                count++;
            }
        }

        if (count == 0)
            return chars;

        float avgWidth = sumWidth / (float)count;
        int sz = newRects.size();

        for (int i = 1; i < sz; i++) {
            int tr = newRects[i - 1].x + newRects[i - 1].width;

            if (newRects[i].x - tr > 0.8 * avgWidth) {
                newRects.push_back(cv::Rect(tr + (int)(0.1 * avgWidth), (newRects[i].y + newRects[i - 1].y) / 2, newRects[i].x - tr + 1 - (int)(0.2 * avgWidth), (int)(1.1 * stdHeight)));
            }
        }
        if(newRects.size() < 7)
        {
            if(newRects[0].x > 1.1 * avgWidth)
            {
                newRects.push_back(cv::Rect(newRects[0].x - (int)(1.1 * avgWidth), newRects[0].y, (int)(1.1 * avgWidth), newRects[0].height));
            }
            if(width > newRects[sz - 1].x + newRects[sz - 1].width + (int)(1.1 * avgWidth))
            {
                newRects.push_back(cv::Rect(newRects[sz -1].x + newRects[sz - 1].width, newRects[sz - 1].y, (int)(1.1 * avgWidth), newRects[sz - 1].height));
            }
        }

        // ==> End 3 ./.
        // 4. Split rectangles that are too long
        std::vector<cv::Rect> finalRects;

        for (uint i = 0; i < newRects.size(); i++) {
            int ratio = std::round((float)newRects[i].width / (1.2 * avgWidth));

            if (ratio > 1) {
                int w = newRects[i].width / ratio;

                for (int j = 0; j < ratio; j++)
                {
                    cv::Rect r(newRects[i].x + (j - 0.1) * w, newRects[i].y, (1.2 * w), newRects[i].height);
                    if(r.x + r.width >= width)
                        r.width = width - r.x;
                    if(r.x < 0)
                    {
                        r.width += r.x;
                        r.x = 0;
                    }
                    finalRects.push_back(r);
                }
            } else {
                finalRects.push_back(newRects[i]);
            }
        }
        newRects.clear();
        std::sort(finalRects.begin(), finalRects.end(), [](cv::Rect a, cv::Rect b) {
            return a.x < b.x;
        });
        float ratio;
        int newWidth, newHeight;
        cv::Rect roi;
        width = input.cols;
        height = input.rows;
        for (uint i = 0; i < finalRects.size(); i++) {
            if(finalRects[i].x + finalRects[i].width > width)
                finalRects[i].width = width - finalRects[i].x;
            if(finalRects[i].y + finalRects[i].height > height)
                finalRects[i].height = height - finalRects[i].y;

            cv::Mat whiteMask(32, 32, CV_8UC1, 255);
//            assert(!input(finalRects[i]).empty());
            cv::Mat c = input(finalRects[i]);
            if(finalRects[i].width < finalRects[i].height)
            {
                ratio = 20.f / (finalRects[i].height);
                newWidth = ratio * finalRects[i].width;
                cv::resize(c, c, cv::Size(newWidth, 20));
                roi.x = (32 - newWidth) / 2;
                roi.y = 6;
                roi.height = 20;
                roi.width = newWidth;
            }
            else
            {
                ratio = 20.f / (finalRects[i].width);
                newHeight = ratio * finalRects[i].height;
                cv::resize(c, c, cv::Size(20, newHeight));
                roi.x = 6;
                roi.y = (32 - newHeight) / 2;
                roi.height = newHeight;
                roi.width = 20;
            }

            c.copyTo(whiteMask(roi));
            chars.push_back(whiteMask.clone());

        }
        finalRects.clear();
        input.release();
        // ==> End split ./.
    }
    return chars;
}

std::vector<cv::Mat> extractCharsSP(cv::Mat input, int *sign)
{
//    std::cout << "Input : " << input.size() << std::endl;
    std::vector<cv::Mat> chars;
    int width = input.cols;
    int height = input.rows;
    float whiteRate;

    if(height < 35)
    {
        return chars;
    }
    // 1. Get popular rectangles
//    std::cout << "stage 1 " << std::endl;
    cv::Mat thresMat = ~input;
    std::vector<std::vector<cv::Point>> contoursPoints;
    std::vector<cv::Vec4i> hierrachy;
    cv::findContours(thresMat, contoursPoints, hierrachy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    thresMat.release();
    int numContoursPoints = contoursPoints.size();
    if (numContoursPoints > 0)
    {
        std::vector<cv::Rect> rects;
        for (int i = 0; i < numContoursPoints; i++)
        {
            cv::Rect rect = cv::boundingRect(contoursPoints[i]);
            // Note: The bounding rectangles may be apart-outside of input image.

            if (rect.height > 12 && rect.width > 3 && rect.width < 0.5 * width && rect.height > 0.25 * height && rect.height < 0.5 * height)
            {
                if(rect.x < 0)
                    rect.x = 0;
                if(rect.x + rect.width > width)
                    rect.width = width - rect.x - 1;
                rects.push_back(rect);
            }
        }

        std::vector<std::pair<float, int>> stats;
        int numRect = rects.size();
        // Group rectangles that have the same height
        if(numRect < 2) // prerequisites in order to have 2 lines at least.
            return chars;
        stats.push_back(std::make_pair((float)rects[0].height, 1));
        float hThres = (float)(0.1 * height);
        for (int i = 1; i < numRect; i++) {
            int numStats = stats.size();
            bool inserted = false;
            for (int j = 0; j < numStats; j++) {
                if (std::abs(stats[j].first - rects[i].height)  < hThres) {
                    stats[j].first = (stats[j].first * stats[j].second + (float)rects[i].height) / (stats[j].second + 1);
                    stats[j].second++;
                    inserted = true;
                    break;
                }
            }

            if (!inserted)
                stats.push_back(std::make_pair((float)rects[i].height, 1));
        }
        int numStats = stats.size();
        int maxCount = stats[0].second;
        int stdHeight = stats[0].first;

        for (int i = 1; i < numStats; i++) {
            if (stats[i].second > maxCount) {
                maxCount = stats[i].second;
                stdHeight = stats[i].first;
            }
        }

        std::vector<cv::Rect> stdRects;

        for (int i = 0; i < numRect; i++) {
            if (std::abs(stdHeight - (float)rects[i].height) < hThres / 2.f + 0.5) {
                // Because rectangle completely bounds a character,
                // when calculating whiteRate, you must shrink the rectangle by 1px
//                assert(!input(cv::Rect(rects[i].x + 1, rects[i].y + 1, rects[i].width - 2, rects[i].height - 2)).empty());
                whiteRate = (float)cv::sum(input(cv::Rect(rects[i].x + 1, rects[i].y + 1, rects[i].width - 2, rects[i].height - 2)))[0] / (float)((rects[i].width - 2) * (rects[i].height - 2) * 255);
                if(whiteRate < 0.75)    // there is no lower bound due to case of number 1.
                    stdRects.push_back(rects[i]);
            }
        }
//        printf("# of stdRects : %d\n", stdRects.size());
//        for(int o = 0; o < stdRects.size(); o++)
//        {
//            std::cout << stdRects[o] << std::endl;
//        }
        if(stdRects.size() < 2)
            return chars;
        rects.clear();
        // ==> End 1 ./.
        // 2. Align lines, remove inner rectangles and extra-get predicted rectangles
//        std::cout << "stage 2 " << std::endl;

        std::sort(stdRects.begin(), stdRects.end(), [](cv::Rect a, cv::Rect b) {
            return a.x < b.x;
        });
        std::vector<cv::Rect> fRects, sRects;
        int fcounter = 0, scounter = -1;
        fRects.push_back(stdRects[0]);

        for(uint i = 1; i < stdRects.size(); i++)
        {
            // Reason: Normally, the first line will be located above the second line
            // Condition 1: Arrange rects to 2 lines
            if(std::abs(stdRects[i].y - stdRects[0].y) < 0.5 * stdHeight)
            {
                // a line
                // Condition 2: Just get no-inner rectangles
                if(stdRects[i].x > fRects[fcounter].x + 0.75 * fRects[fcounter].width && stdRects[i].x + stdRects[i].width > fRects[fcounter].x + fRects[fcounter].width)
                {
                    // Condition 3: Get extra-rectangles
                    if(stdRects[i].x - fRects[fcounter].x - fRects[fcounter].width > 0.5 * stdHeight)
                    {
                        cv::Rect r(fRects[fcounter].x + fRects[fcounter].width,
                                   fRects[fcounter].y < stdRects[i].y ? fRects[fcounter].y : stdRects[i].y,
                                   stdRects[i].x - fRects[fcounter].x - fRects[fcounter].width,
                                   stdHeight + std::abs(stdRects[i].y - fRects[fcounter].y));
						r.x = r.x < 0 ? 0 : r.x;
						r.y = r.y < 0 ? 0 : r.y;
						r.width = r.x + r.width - 1 >= width ? width - r.x : r.width;
						r.height = r.y + r.height - 1 >= height ? height - r.y : r.height;
                        cv::Mat region = ~input(r);
                        whiteRate = (float)cv::sum(region)[0] / (float)(region.rows * region.cols * 255);
                        if(whiteRate > 0.1 && whiteRate < 0.9)
                        {
                            std::vector<std::vector<cv::Point>> contoursPoints0;
                            std::vector<cv::Vec4i> hierrachy0;
                            cv::findContours(region, contoursPoints0, hierrachy0, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
                            region.release();
                            if(contoursPoints0.size() > 0)
                            {
                                bool had = false;
                                for(uint j = 0; j < contoursPoints0.size(); j++)
                                {
                                    cv::Rect roi = cv::boundingRect(contoursPoints0[j]);
                                    if(roi.width > 3 && roi.height > 0.75 * r.height)
                                    {
                                        fcounter++;
                                        roi.x += r.x;
                                        roi.y += r.y;
                                        fRects.push_back(roi);
                                        had = true;
                                    }
                                }
                                if(!had)
                                {
                                    r.x += (int)(0.1 * r.height);
                                    r.width -= (int)(0.2 * r.height);
                                    fRects.push_back(r);
                                    fcounter++;
                                }
                            }
                        }
                    }
                    fRects.push_back(stdRects[i]);
                    fcounter++;
                }
            }
            else
            {
                // another line
                if(scounter == -1)
                {
                    sRects.push_back(stdRects[i]);
                    scounter++;
                }
                else
                {
                    // Condition 2: Just get no-inner rectangles
                    if(stdRects[i].x > sRects[scounter].x + 0.75 * sRects[scounter].width && stdRects[i].x + stdRects[i].width > sRects[scounter].x + sRects[scounter].width)
                    {
                        // Condition 3: Get extra-rectangles
                        if(stdRects[i].x - sRects[scounter].x - sRects[scounter].width > 0.5 * stdHeight)
                        {
                            cv::Rect r(sRects[scounter].x + sRects[scounter].width,
                                       sRects[scounter].y < stdRects[i].y ? sRects[scounter].y : stdRects[i].y,
                                       stdRects[i].x - sRects[scounter].x - sRects[scounter].width,
                                       stdHeight + std::abs(stdRects[i].y - sRects[scounter].y));
                            r.x = r.x < 0 ? 0 : r.x;
                            r.y = r.y < 0 ? 0 : r.y;
                            r.width = r.x + r.width - 1 >= width ? width - r.x : r.width;
                            r.height = r.y + r.height - 1 >= height ? height - r.y : r.height;
//                            assert(!input(r).empty());
                            cv::Mat region = ~input(r);
                            whiteRate = (float)cv::sum(region)[0] / (float)(region.rows * region.cols * 255);
                            if(whiteRate > 0.1 && whiteRate < 0.9)
                            {
                                std::vector<std::vector<cv::Point>> contoursPoints0;
                                std::vector<cv::Vec4i> hierrachy0;
                                cv::findContours(region, contoursPoints0, hierrachy0, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
                                region.release();
                                if(contoursPoints0.size() > 0)
                                {
                                    bool had = false;
                                    for(uint j = 0; j < contoursPoints0.size(); j++)
                                    {
                                        cv::Rect roi = cv::boundingRect(contoursPoints0[j]);
                                        if(roi.width > 3 && roi.height > 0.75 * r.height)
                                        {
                                            scounter++;
                                            roi.x += r.x;
                                            roi.y += r.y;
                                            sRects.push_back(roi);
                                            had = true;
                                        }
                                    }
                                    if(!had)
                                    {
                                        r.x += (int)(0.1 * r.height);
                                        r.width -= (int)(0.2 * r.height);
                                        sRects.push_back(r);
                                        scounter++;
                                    }
                                }
                            }
                        }
                        sRects.push_back(stdRects[i]);
                        scounter++;
                    }
                }
            }
        }

        std::sort(fRects.begin(), fRects.end(), [](cv::Rect a, cv::Rect b) {
            return a.x < b.x;
        });
        std::sort(sRects.begin(), sRects.end(), [](cv::Rect a, cv::Rect b) {
            return a.x < b.x;
        });
        // End 2 ./.
        // 3. Arrange lines and remove boder noise
//        std::cout << "stage 3 " << std::endl;

        if(fRects.size() < 1 || sRects.size() < 1)
            return chars;
        std::vector<cv::Rect> firstLine, secondLine;
        // Identify the landmarks in location
        int sStd = sRects.size() / 2;
        int fStd = fRects.size() / 2;
        // Arrange order of the lines
        if(fRects[0].y < sRects[0].y)
        {
            for(uint i = 0; i < fRects.size(); i++)
            {
                if(std::abs(fRects[i].y - sRects[sStd].y) > (int)(0.7 * sRects[sStd].height))
                {
                    if(i == 0)
                        firstLine.push_back(fRects[i]);
                    else
                    {
                        // Check innner condition
                        if(fRects[i].x > fRects[i - 1].x + 0.75 * fRects[i - 1].width && fRects[i].x + fRects[i].width > fRects[i - 1].x + fRects[i - 1].width)
                        {
                            firstLine.push_back(fRects[i]);
                        }
                    }
                }
            }
            for(uint i = 0; i < sRects.size(); i++)
            {
                if(std::abs(sRects[i].y - fRects[fStd].y) > (int)(0.7 * fRects[fStd].height))
                {
                    if(i == 0)
                    {
                        secondLine.push_back(sRects[i]);
//                        std::cout << sRects[i] << std::endl;
                    }
                    else
                    {
                        // Check innner condition
                        if(sRects[i].x > sRects[i - 1].x + 0.75 * sRects[i - 1].width && sRects[i].x + sRects[i].width > sRects[i - 1].x + sRects[i - 1].width)
                        {
                            secondLine.push_back(sRects[i]);
//                            std::cout << sRects[i] << std::endl;
                        }
                    }
                }
            }
        }
        else
        {
            for(uint i = 0; i < fRects.size(); i++)
            {
                if(std::abs(fRects[i].y - sRects[sStd].y) > (int)(0.7 * sRects[sStd].height))
                {
                    if(i == 0)
                    {
                        secondLine.push_back(fRects[i]);
//                        std::cout << fRects[i] << std::endl;
                    }
                    else
                    {
                        // Check innner condition
                        if(fRects[i].x > fRects[i - 1].x + 0.75 * fRects[i - 1].width && fRects[i].x + fRects[i].width > fRects[i - 1].x + fRects[i - 1].width)
                        {
                            secondLine.push_back(fRects[i]);
//                            std::cout << fRects[i] << std::endl;
                        }
                    }
                }
            }
            for(uint i = 0; i < sRects.size(); i++)
            {
                if(std::abs(sRects[i].y - fRects[fStd].y) > (int)(0.7 * fRects[fStd].height))
                {
                    if(i == 0)
                        firstLine.push_back(sRects[i]);
                    else
                    {
                        // Check innner condition
                        if(sRects[i].x > sRects[i - 1].x + 0.75 * sRects[i - 1].width && sRects[i].x + sRects[i].width > sRects[i - 1].x + sRects[i - 1].width)
                        {
                            firstLine.push_back(sRects[i]);
                        }
                    }
                }
            }
        }
//        for(int o = 0; o < fRects.size(); o++)
//            std::cout << fRects[o] << std::endl;
//        for(int o = 0; o < sRects.size(); o++)
//        {
//            std::cout << sRects[o] << std::endl;
//        }
        fRects.clear();
        sRects.clear();
        // 4. Split too long rectangles
//        std::cout << "stage 4 " << std::endl;

        std::vector<cv::Rect> fLine, sLine;
        // Calculate ratio of height to width
        float hpw = 1.7;
        for(uint i = 0; i < secondLine.size(); i++)
        {
            if((float)secondLine[i].height / (float)secondLine[i].width <= 2.6 &&
               (float)secondLine[i].height / (float)secondLine[i].width > 1.6)
            {
                hpw = (float)secondLine[i].height / (float)secondLine[i].width;
                break;
            }
        }
//        std::cout << "stage 4.1 " << std::endl;
        for(uint i = 0; i < firstLine.size(); i++)
        {
            float ratio = (float)firstLine[i].height / (float)firstLine[i].width;
            if(hpw / ratio > 1.9)
            {
                // Too long
                int numSeg = std::round(hpw / ratio / 1.1);
                int w = std::ceil(firstLine[i].width / numSeg);
                for(int j = 0; j < numSeg; j++)
                {
                    cv::Rect r(firstLine[i].x + w * j, firstLine[i].y, w + 1, firstLine[i].height);
                    r.x = r.x < 0 ? 0 : r.x;
                    r.y = r.y < 0 ? 0 : r.y;
                    r.width = r.x + r.width - 1 >= width ? width - r.x : r.width;
                    r.height = r.y + r.height - 1 >= height ? height - r.y : r.height;
//                    assert(!input(r).empty());
                    whiteRate = (float)cv::sum(input(r))[0] / (float)(r.width * r.height * 255);
                    if(whiteRate > 0.1 && whiteRate < 0.9)
                        fLine.push_back(r);
                }
            }
            else
                fLine.push_back(firstLine[i]);
        }
//        std::cout << "stage 4.2 " << std::endl;
        for(uint i = 0; i < secondLine.size(); i++)
        {
            float ratio = (float)secondLine[i].height / (float)secondLine[i].width;
//            assert(!input(secondLine[i]).empty());
            if(hpw / ratio > 1.9)
            {
                // Too long
                int numSeg = std::round(hpw / ratio / 1.1);
                int w = std::ceil(secondLine[i].width / numSeg);
//                std::cout << "second line i" << secondLine[i] << std::endl;
//                std::cout << "# of segments : " << numSeg << std::endl;
                for(int j = 0; j < numSeg; j++)
                {
                    cv::Rect r(secondLine[i].x + w * j, secondLine[i].y, w + 1, secondLine[i].height);
                    r.x = r.x < 0 ? 0 : r.x;
                    r.y = r.y < 0 ? 0 : r.y;
                    r.width = r.x + r.width - 1 >= width ? width - r.x : r.width;
                    r.height = r.y + r.height - 1 >= height ? height - r.y : r.height;
//                    std::cout << "input size: " << input.size() << "\t R: x" << r.x << " y" << r.y << "w " << r.width << "h " << r.height << std::endl;
//                    assert(!input(r).empty());
                    whiteRate = (float)cv::sum(input(r))[0] / (float)(r.width * r.height * 255);
                    if(whiteRate > 0.1 && whiteRate < 0.9)
                        sLine.push_back(r);
                }
            }
            else
            {
                sLine.push_back(secondLine[i]);
            }
        }
        firstLine.clear();
        secondLine.clear();
        if(fLine.size() == 0 || sLine.size() == 0)
            return chars;

        // End 4 ./.
        // 5. Expand 2 sides
//        std::cout << "stage 5 " << std::endl;

        int fLineSize = fLine.size();
        cv::Rect testRect;

        // Expand for the first line
        // On the left
        cv::Rect mark = fLine[0];
        int w = std::round(mark.height / hpw);
        int dw = (int)(0.12 * mark.height);
        while(true)
        {
            if(mark.x >= (float)w / 2.f + (float)dw)
            {
                testRect.x = (int)(mark.x - (float)w / 2.f - (float)dw);
                testRect.y = mark.y;
                testRect.width = w / 2;
                testRect.height = stdHeight;
                testRect.x = testRect.x < 0 ? 0 : testRect.x;
                testRect.y = testRect.y < 0 ? 0 : testRect.y;
                testRect.width = testRect.x + testRect.width - 1 >= width ? width - testRect.x : testRect.width;
                testRect.height = testRect.y + testRect.height - 1 >= height ? height - testRect.y : testRect.height;
                whiteRate = (float)cv::sum(input(testRect))[0] / (float)(testRect.width * testRect.height * 255);
                if(whiteRate > 0.1 && whiteRate < 0.89)
                {
                    testRect.x = mark.x - w - dw < 0 ? 0 : mark.x - w - dw;
                    testRect.width = mark.x - dw - testRect.x + 1;
                    testRect.height = mark.height;
                    whiteRate = (float)cv::sum(input(testRect))[0] / (float)(testRect.width * testRect.height * 255);
                    if(whiteRate > 0.1 && whiteRate < 0.9)
                    {
                        mark = testRect;
                        fLine.push_back(testRect);
                    }
                    else
                        break;
                }
                else
                    break;
            }
            else
                break;
        }

        // On the right
        mark = fLine[fLineSize - 1];
        w = std::round(mark.height / hpw);
        dw = (int)(0.12 * mark.height);
        while(true)
        {
            if(mark.x + mark.width <= (float)width - (float)w / 2.f - (float)dw)
            {
                testRect.x = mark.x + mark.width + dw;
                testRect.y = mark.y;
                testRect.width = w / 2;
                testRect.height = stdHeight;
                testRect.x = testRect.x < 0 ? 0 : testRect.x;
                testRect.y = testRect.y < 0 ? 0 : testRect.y;
                testRect.width = testRect.x + testRect.width - 1 >= width ? width - testRect.x : testRect.width;
                testRect.height = testRect.y + testRect.height - 1 >= height ? height - testRect.y : testRect.height;
                whiteRate = (float)cv::sum(input(testRect))[0] / (float)(testRect.width * testRect.height * 255);
                if(whiteRate > 0.1 && whiteRate < 0.89)
                {
                    testRect.width = testRect.x + w - 1 < width ? w : width - testRect.x;
                    testRect.height = mark.height;
                    whiteRate = (float)cv::sum(input(testRect))[0] / (float)(testRect.width * testRect.height * 255);
                    if(whiteRate > 0.1 && whiteRate < 0.9)
                    {
                        mark = testRect;
                        fLine.push_back(testRect);
                    }
                    else
                        break;
                }
                else
                    break;
            }
            else
                break;
        }
        std::sort(fLine.begin(), fLine.end(), [](cv::Rect a, cv::Rect b)
        {
            return a.x < b.x;
        });

        // Expand for the second line
        // On the left
        int sLineSize = sLine.size();
        mark = sLine[0];
        w = std::round(mark.height / hpw);
        dw = (int)(0.12 * mark.height);
        while(true)
        {
            if(mark.x >= (float)w / 2.f + (float)dw)
            {
                testRect.x = (int)(mark.x - (float)w / 2.f - (float)dw);
                testRect.y = mark.y;
                testRect.width = w / 2;
                testRect.height = stdHeight;
                testRect.x = testRect.x < 0 ? 0 : testRect.x;
                testRect.y = testRect.y < 0 ? 0 : testRect.y;
                testRect.width = testRect.x + testRect.width - 1 >= width ? width - testRect.x : testRect.width;
                testRect.height = testRect.y + testRect.height - 1 >= height ? height - testRect.y : testRect.height;
                whiteRate = (float)cv::sum(input(testRect))[0] / (float)(testRect.width * testRect.height * 255);
                if(whiteRate > 0.1 && whiteRate < 0.89)
                {
                    testRect.x = mark.x - w - dw < 0 ? 0 : mark.x - w - dw;
                    testRect.width = mark.x - dw - testRect.x + 1;
                    testRect.height = mark.height;
                    whiteRate = (float)cv::sum(input(testRect))[0] / (float)(testRect.width * testRect.height * 255);
                    if(whiteRate > 0.1 && whiteRate < 0.9)
                    {
                        mark = testRect;
                        sLine.push_back(testRect);
                    }
                    else
                        break;
                }
                else
                    break;
            }
            else
                break;
        }

        // On the right
        mark = sLine[sLineSize - 1];
        w = std::round(mark.height / hpw);
        dw = (int)(0.12 * mark.height);
        while(true)
        {
            if(mark.x + mark.width <= (float)width - (float)w / 2.f - (float)dw)
            {
                testRect.x = mark.x + mark.width + dw;
                testRect.y = mark.y;
                testRect.width = w / 2;
                testRect.height = stdHeight;
                testRect.x = testRect.x < 0 ? 0 : testRect.x;
                testRect.y = testRect.y < 0 ? 0 : testRect.y;
                testRect.width = testRect.x + testRect.width - 1 >= width ? width - testRect.x : testRect.width;
                testRect.height = testRect.y + testRect.height - 1 >= height ? height - testRect.y : testRect.height;
                whiteRate = (float)cv::sum(input(testRect))[0] / (float)(testRect.width * testRect.height * 255);
                if(whiteRate > 0.1 && whiteRate < 0.89)
                {
                    testRect.width = testRect.x + w - 1 < width ? w : width - testRect.x;
                    testRect.height = mark.height;
                    whiteRate = (float)cv::sum(input(testRect))[0] / (float)(testRect.width * testRect.height * 255);
                    if(whiteRate > 0.1 && whiteRate < 0.9)
                    {
                        mark = testRect;
                        sLine.push_back(testRect);
                    }
                    else
                        break;
                }
                else
                    break;
            }
            else
                break;
        }
        std::sort(sLine.begin(), sLine.end(), [](cv::Rect a, cv::Rect b)
        {
            return a.x < b.x;
        });

        // End 5 ./.
        // Show rectangles
//        std::cout << "stage 6 " << std::endl;

        *sign = fLine.size();
        float ratio;
        cv::Rect roi;
        int newWidth, newHeight;
        for(uint i = 0; i < fLine.size(); i++)
        {
            int dh = std::ceil(0.1 * fLine[i].height);
            int dw = std::ceil(0.1 * fLine[i].width);
            if(fLine[i].x >= dw)
            {
                fLine[i].x -= dw;
                fLine[i].width += dw;
            }
            if((fLine[i].x + fLine[i].width + dw) < width - 1)
            {
                fLine[i].width += dw;
            }
            if(fLine[i].y >= dh)
            {
                fLine[i].y -= dh;
                fLine[i].height += 2 * dh;
            }
//            assert(!input(fLine[i]).empty());
            cv::Mat c = input(fLine[i]);
            cv::Mat whiteMask(32, 32, CV_8UC1, 255);
            if(fLine[i].width < fLine[i].height)
            {
                ratio = 20.f / (fLine[i].height);
                newWidth = ratio * fLine[i].width;
                cv::resize(c, c, cv::Size(newWidth, 20));
                roi.x = (32 - newWidth) / 2;
                roi.y = 6;
                roi.height = 20;
                roi.width = newWidth;
            }
            else
            {
                ratio = 20.f / (fLine[i].width);
                newHeight = ratio * fLine[i].height;
                cv::resize(c, c, cv::Size(20, newHeight));
                roi.x = 6;
                roi.y = (32 - newHeight) / 2;
                roi.height = newHeight;
                roi.width = 20;
            }

            c.copyTo(whiteMask(roi));
            chars.push_back(whiteMask);
        }
        for(uint i = 0; i < sLine.size(); i++)
        {
            int dh = std::ceil(0.1 * sLine[i].height);
            int dw = std::ceil(0.1 * sLine[i].width);
            if(sLine[i].x >= dw)
            {
                sLine[i].x -= dw;
                sLine[i].width += dw;
            }
            if(sLine[i].x + sLine[i].width + dw <= width)
            {
                sLine[i].width += dw;
            }
            sLine[i].y -= dh;
            sLine[i].height += dh;
            if(sLine[i].y + sLine[i].height + dh <= height)
            {
                sLine[i].height += dh;
            }
//            assert(!input(sLine[i]).empty());
            cv::Mat c = input(sLine[i]);
            cv::Mat whiteMask(32, 32, CV_8UC1, 255);
            if(sLine[i].width < sLine[i].height)
            {
                ratio = 20.f / (sLine[i].height);
                newWidth = std::ceil(ratio * sLine[i].width);
                cv::resize(c, c, cv::Size(newWidth, 20));
                roi.x = (32 - newWidth) / 2;
                roi.y = 6;
                roi.height = 20;
                roi.width = newWidth;
            }
            else
            {
                ratio = 20.f / (sLine[i].width);
                newHeight = std::ceil(ratio * sLine[i].height);
                cv::resize(c, c, cv::Size(20, newHeight));
                roi.x = 6;
                roi.y = (32 - newHeight) / 2;
                roi.height = newHeight;
                roi.width = 20;
            }
            c.copyTo(whiteMask(roi));
            chars.push_back(whiteMask);
        }
    }
//    std::cout << "stage 7 " << std::endl;

    return chars;
}

std::vector<cv::Mat> extractChars(cv::Mat input, int *sign)
{
    std::vector<cv::Mat> chars;
    int width = input.cols;
    int height = input.rows;

    if(height < 35)
    {
        return chars;
    }
    cv::Mat thresMat = ~input;
    std::vector<std::vector<cv::Point>> contoursPoints;
    std::vector<cv::Vec4i> hierrachy;
    cv::findContours(thresMat, contoursPoints, hierrachy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    thresMat.release();
    int numContoursPoints = contoursPoints.size();
    if (numContoursPoints > 0)
    {
        std::vector<cv::Rect> rects;
        for (int i = 0; i < numContoursPoints; i++)
        {
            cv::Rect rect = cv::boundingRect(contoursPoints[i]);

            if (rect.height > 12 && rect.width > 3 && rect.width < 0.5 * width && rect.height < 0.5 * height)
            {
                rects.push_back(rect);
            }
        }
        std::vector<std::pair<float, int>> stats;
        int numRect = rects.size();
        // 1. Group rectangles that have the same height
        if(numRect < 2)
            return chars;
        stats.push_back(std::make_pair((float)rects[0].height, 1));
        float hThres = (float)(0.1 * height);
        for (int i = 1; i < numRect; i++) {
            int numStats = stats.size();
            bool inserted = false;
            for (int j = 0; j < numStats; j++) {
                if (std::abs(stats[j].first - rects[i].height)  < hThres) {
                    stats[j].first = (stats[j].first * stats[j].second + (float)rects[i].height) / (stats[j].second + 1);
                    stats[j].second++;
                    inserted = true;
                    break;
                }
            }

            if (!inserted)
                stats.push_back(std::make_pair((float)rects[i].height, 1));
        }
        int numStats = stats.size();
        int maxCount = stats[0].second;
        int stdHeight = stats[0].first;

        for (int i = 1; i < numStats; i++) {
            if (stats[i].second > maxCount) {
                maxCount = stats[i].second;
                stdHeight = stats[i].first;
            }
        }
        std::vector<cv::Rect> stdRects;

        for (int i = 0; i < numRect; i++) {
            if (std::abs(stdHeight - (float)rects[i].height) < hThres / 2.f + 0.5) {
                stdRects.push_back(rects[i]);
            }
        }
        rects.clear();
        // ==> End 1 ./.
        // 2. Remove inner rectangles and extra-get predicted rectangles
        std::sort(stdRects.begin(), stdRects.end(), [](cv::Rect a, cv::Rect b) {
            return a.x < b.x;
        });
        std::vector<cv::Rect> fRects, sRects;
        int fcounter = 0, scounter = -1;
        fRects.push_back(stdRects[0]);

        for(uint i = 1; i < stdRects.size(); i++)
        {
            if(std::abs(stdRects[i].y - stdRects[0].y) < 0.5 * stdHeight)
            {
                if(stdRects[i].x > fRects[fcounter].x + 0.75 * fRects[fcounter].width && stdRects[i].x + stdRects[i].width > fRects[fcounter].x + fRects[fcounter].width)
                {
                    if(stdRects[i].x - fRects[fcounter].x - fRects[fcounter].width > 0.5 * stdHeight)
                    {
                        cv::Rect r(fRects[fcounter].x + fRects[fcounter].width, stdRects[i].y < fRects[fcounter].y ? stdRects[i].y : fRects[fcounter].y, stdRects[i].x - fRects[fcounter].x - fRects[fcounter].width, (int)(1 * stdHeight)/*fRects[fcounter].height + std::abs(stdRects[i].y - fRects[fcounter].y)*/);

                        r.x += (int)(0.1 * r.height);
                        r.width -= (int)(0.2 * r.height);
                        fRects.push_back(r);
                        fcounter++;
                    }
                    fRects.push_back(stdRects[i]);
                    fcounter++;
                }

            }
            else
            {
                if(scounter == -1)
                {
                    sRects.push_back(stdRects[i]);
                    scounter++;
                }
                else
                {
                    if(stdRects[i].x > sRects[scounter].x + 0.75 * sRects[scounter].width && stdRects[i].x + stdRects[i].width > sRects[scounter].x + sRects[scounter].width)
                    {
                        if(stdRects[i].x - sRects[scounter].x - sRects[scounter].width > 0.5 * stdHeight)
                        {
                            cv::Rect r(sRects[scounter].x + sRects[scounter].width, stdRects[i].y < sRects[scounter].y ? stdRects[i].y : sRects[scounter].y, stdRects[i].x - sRects[scounter].x - sRects[scounter].width, stdHeight/*sRects[scounter].height + std::abs(stdRects[i].y - sRects[scounter].y)*/);
                            r.x += (int)(0.1 * r.height);
                            r.width -= (int)(0.2 * r.height);
                            sRects.push_back(r);
                            scounter++;
                        }
                        sRects.push_back(stdRects[i]);
                        scounter++;
                    }
                }
            }

        }

        // End 2 ./.
        // 3. Arrange lines and remove boder noise
        if(fRects.size() < 1 || sRects.size() < 1)
            return chars;
        std::vector<cv::Rect> firstLine, secondLine;
        int sStd = sRects.size() / 2;
        int fStd = fRects.size() / 2;

        if(fRects[0].y < sRects[0].y)
        {
            for(uint i = 0; i < fRects.size(); i++)
            {
                    firstLine.push_back(fRects[i]);
            }
            for(uint i = 0; i < sRects.size(); i++)
            {
                    secondLine.push_back(sRects[i]);
            }
        }
        else
        {
            for(uint i = 0; i < fRects.size(); i++)
            {
                if(std::abs(fRects[i].y - sRects[sStd].y) > (int)(0.7 * sRects[sStd].height))
                    secondLine.push_back(fRects[i]);
            }
            for(uint i = 0; i < sRects.size(); i++)
            {
                if(std::abs(sRects[i].y - fRects[fStd].y) > (int)(0.7 * fRects[fStd].height))
                    firstLine.push_back(sRects[i]);
            }
        }
        fRects.clear();
        sRects.clear();

        // 4. Split too long rectangles
        if(firstLine.size() < 1 || secondLine.size() < 1)
            return chars;
        std::vector<cv::Rect> fLine = firstLine, sLine = secondLine;
        // End 4 ./.
        // 6. Extract characters
        *sign = fLine.size();
        float ratio;
        cv::Rect roi;
        int newWidth, newHeight;
        for(uint i = 0; i < fLine.size(); i++)
        {
            int dh = std::ceil(0.1 * fLine[i].height);
            int dw = std::ceil(0.1 * fLine[i].width);
            if(fLine[i].x >= dw)
            {
                fLine[i].x -= dw;
                fLine[i].width += dw;
            }
            if(fLine[i].x + fLine[i].width + dw <= width)
            {
                fLine[i].width += dw;
            }
            if(fLine[i].y >= dh)
            {
                fLine[i].y -= dh;
                fLine[i].height += 2 * dh;
                if(fLine[i].y + fLine[i].height > height)
                    fLine[i].height -= dh;
            }
            cv::Mat c = input(fLine[i]);
            cv::Mat whiteMask(32, 32, CV_8UC1, 255);
            if(fLine[i].width < fLine[i].height)
            {
                ratio = 20.f / (fLine[i].height);
                newWidth = ratio * fLine[i].width;
                cv::resize(c, c, cv::Size(newWidth, 20));
                roi.x = (32 - newWidth) / 2;
                roi.y = 6;
                roi.height = 20;
                roi.width = newWidth;
            }
            else
            {
                ratio = 20.f / (fLine[i].width);
                newHeight = ratio * fLine[i].height;
                cv::resize(c, c, cv::Size(20, newHeight));
                roi.x = 6;
                roi.y = (32 - newHeight) / 2;
                roi.height = newHeight;
                roi.width = 20;
            }
            c.copyTo(whiteMask(roi));
            chars.push_back(whiteMask);
        }
        for(uint i = 0; i < sLine.size(); i++)
        {
            int dh = std::ceil(0.1 * sLine[i].height);
            int dw = std::ceil(0.1 * sLine[i].width);
            if(sLine[i].x >= dw)
            {
                sLine[i].x -= dw;
                sLine[i].width += dw;
            }
            if(sLine[i].x + sLine[i].width + dw <= width)
            {
                sLine[i].width += dw;
            }
            if(sLine[i].y >= dh)
            {
                sLine[i].y -= dh;
                sLine[i].height += dh;
            }

            if(sLine[i].y + sLine[i].height + dh <= height)
            {
                sLine[i].height += dh;
            }
            cv::Mat c = input(sLine[i]);
            cv::Mat whiteMask(32, 32, CV_8UC1, 255);
            if(sLine[i].width < sLine[i].height)
            {
                ratio = 20.f / (sLine[i].height);
                newWidth = std::ceil(ratio * sLine[i].width);
                cv::resize(c, c, cv::Size(newWidth, 20));
                roi.x = (32 - newWidth) / 2;
                roi.y = 6;
                roi.height = 20;
                roi.width = newWidth;
            }
            else
            {
                ratio = 20.f / (sLine[i].width);
                newHeight = std::ceil(ratio * sLine[i].height);
                cv::resize(c, c, cv::Size(20, newHeight));
                roi.x = 6;
                roi.y = (32 - newHeight) / 2;
                roi.height = newHeight;
                roi.width = 20;
            }
            c.copyTo(whiteMask(roi));
            chars.push_back(whiteMask);
        }
    }

    return chars;
}
/*
cv::Mat deskewImage(cv::Mat image)
{
    if(image.channels() > 1)
    {
        std::vector<cv::Mat> channels;
        cv::Mat hsvImg ;
        cv::cvtColor(image, hsvImg, CV_RGB2HSV);
        cv::split(hsvImg, channels);
        image = channels[2].clone();
        hsvImg.release();
        channels.clear();
    }
    cv::Mat enhanced(image.size(), CV_8UC1);
    contrastEnhance(image, enhanced);
    cv::Mat thresh1;
    cv::threshold(enhanced, thresh1, 90, 255, cv::THRESH_OTSU);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    //    thresh1 = ~thresh1;
    cv::findContours(thresh1, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
//    cv::Mat contourMap = cv::Mat::zeros(cv::Size(thresh1.cols, thresh1.rows), CV_8U);
    int bigestContourIdx = -1;
    float bigestContourArea = 0;
    cv::Rect ctBox;
    float ctArea;
//    std::vector<std::vector<cv::Point>> charcontours;

    for (int i = 0; i < contours.size(); i++) {
        ctArea = cv::contourArea(contours[i]);

        if (ctArea < 30) {
            continue;
        }

        ctBox = cv::boundingRect(contours[i]);

        if (ctArea > bigestContourArea) {
            bigestContourArea = ctArea;
            bigestContourIdx = i;
        }
    }

    cv::Mat plgray;

    if (bigestContourIdx > 0) {
        cv::RotatedRect boundingBox = cv::minAreaRect(contours[bigestContourIdx]);
        float angle = boundingBox.angle;

        if (angle <= -45.0 && angle >= -90.0) {
            angle = 90.0 + angle;
        }

        if (angle >= 90.0) {
            angle = angle - 90.0;
        }

        if (angle >= 45.0 && angle <= 90.0) {
            angle = 90.0 - angle;
        }

        if (abs(angle) > 4) {
            cv::Point2f center = cv::Point2f((float)thresh1.cols / 2.0, (float)thresh1.rows / 2.0);
            cv::Mat R = cv::getRotationMatrix2D(center, angle, 1.0);
            cv::warpAffine(image, image, R, thresh1.size(), cv::INTER_CUBIC);
            //            cout << " size of sub [ W - H ]" << (int)boundingBox.size.width << "-" <<
            //                    boundingBox.size.height << endl;
            float ratio = boundingBox.size.width / boundingBox.size.height;

            if (ratio < 0.8) {
                cv::getRectSubPix(image, cv::Size((int)boundingBox.size.height, (int)boundingBox.size.width),
                                  boundingBox.center, plgray);
            } else {
                cv::getRectSubPix(image, cv::Size((int)boundingBox.size.width, (int)boundingBox.size.height),
                                  boundingBox.center, plgray);
            }
        } else plgray = image.clone();
    } else plgray = image.clone();

    return plgray;
}
*/
void binarize(const cv::Mat input, cv::Mat &output, const int windowSize)
{
    int paddingSize = windowSize / 2;
    cv::Mat paddedImg;
    insertPadding(input, paddedImg, paddingSize, 127);
    int height      = paddedImg.rows;
    int width       = paddedImg.cols;
    output          = cv::Mat(input.size(), CV_8UC1, 255);
    cv::Mat thress  = cv::Mat::zeros(input.size(), CV_32FC1);
    cv::parallel_for_(cv::Range(0, (width - windowSize + 1) * (height - windowSize + 1)), [&](const cv::Range & range) {
        for (int k = range.start; k < range.end; k++) {
            int i = k / (height - windowSize + 1);
            int j = k % (height - windowSize + 1);
            cv::Scalar mean, stdev;
            cv::Rect roi(i, j , windowSize, windowSize);
            cv::meanStdDev(paddedImg(roi), mean, stdev);

            if (stdev[0] < 6.f)
                thress.at<float>(j, i) = 0.8 * (mean[0] + 2.0 * stdev[0]);
            else
                thress.at<float>(j, i) = mean[0] * (1.f + stdev[0] / 1024.f);
        }
    });
    cv::Mat temp;
    paddedImg(cv::Rect(paddingSize, paddingSize, input.cols, input.rows)).copyTo(temp);
    temp.convertTo(temp, CV_32FC1);
    cv::compare(temp, thress, output, cv::CMP_GE);
}

void filter(const cv::Mat input, cv::Mat &output)
{
    int filterSize = 3;
    filterBlobNoise(input, output, filterSize);
}


//-----------------------------/-
//===============================
// II. Support Functions        "
//===============================
void contrastEnhance(cv::Mat &src, cv::Mat &dst, int dist)
{
    cv::Mat smooth;
    cv::GaussianBlur(src, smooth, cv::Size(0, 0), 3);
    int a, b;
    int val, smt;

    for (int x = 0; x < src.cols; x++)
        for (int y = 0 ; y < src.rows; y++) {
            val = (int) src.at<uchar>(y, x);
            smt = (int) smooth.at<uchar>(y, x);

            if ((val - smt) > dist) smt = smt + (val - smt) * 0.5;

            smt = smt < 0.5 * dist ? 0.5 * dist : smt;
            b = smt + 0.5 * dist;
            b = b > 255 ? 255 : b;
            a = b - dist;
            a = a < 0 ? 0 : a;

            if (val >= a && val <= b) {
                dst.at<uchar>(y, x) = (int)(((val - a) / (0.5 * dist)) * 255);
            } else if (val < a) {
                dst.at<uchar>(y, x) = 0;
            } else if (val > b) {
                dst.at<uchar>(y, x) = 255;
            }
        }
}

void insertPadding(const cv::Mat input, cv::Mat &output, const int paddingSize, const int paddingValue)
{
    int width = input.cols;
    int height = input.rows;
    output = cv::Mat(height + 2 * paddingSize, width + 2 * paddingSize, input.type(), paddingValue);
    cv::Rect roi(paddingSize, paddingSize, width, height);
    input.copyTo(output(roi));
}


void filterBlobNoise(const cv::Mat input, cv::Mat &output, const int windowSize)
{
    int paddingSize = windowSize / 2;
    cv::Mat paddedImg;
    insertPadding(input, paddedImg, paddingSize, 255);
    output = paddedImg.clone();
    int rows = paddedImg.rows;
    int cols = paddedImg.cols;

    for (int i = 0; i <= rows - 2 * paddingSize; i++)
        for (int j = 0; j <= cols - 2 * paddingSize; j++)
            if (paddedImg.at<uchar>(i + paddingSize, j + paddingSize) == 0)
                if (paddedImg.at<uchar>(i + paddingSize - 1, j + paddingSize - 1) +
                    paddedImg.at<uchar>(i + paddingSize - 1, j + paddingSize) +
                    paddedImg.at<uchar>(i + paddingSize - 1, j + paddingSize + 1) +
                    paddedImg.at<uchar>(i + paddingSize, j + paddingSize - 1) +
                    paddedImg.at<uchar>(i + paddingSize, j + paddingSize) +
                    paddedImg.at<uchar>(i + paddingSize, j + paddingSize + 1) +
                    paddedImg.at<uchar>(i + paddingSize + 1, j + paddingSize - 1) +
                    paddedImg.at<uchar>(i + paddingSize + 1, j + paddingSize) +
                    paddedImg.at<uchar>(i + paddingSize + 1, j + paddingSize + 1) >= 1020)
                    output.at<uchar>(i + paddingSize, j + paddingSize) = 255;
}

void horiPrune(const cv::Mat input, cv::Mat &output, float thres)
{
    int width = input.cols;
    int height = input.rows;
    int counter;
    std::vector<int> vertHisto;

    for (int i = 0; i < height; i++) {
        counter = 0;

        for (int j = 0; j < width; j++) {
            if (input.at<uchar>(i, j) == 0)
                counter++;
        }

        vertHisto.push_back(counter);
    }

    cv::Rect hroi(0, 0, width, height);
    std::vector<cv::Point> ranges;
    cv::Point range;
    gState state = WAITING;
    int threshold = thres * width;

    for (int i = 0; i < height; i++) {
        if (state == WAITING) {
            if (vertHisto[i] > threshold) {
                state = COUNTING;
                range.x = i;
            }
        } else if (state == COUNTING) {
            if (vertHisto[i] <= threshold) {
                state = WAITING;
                range.y = i - 1;
                ranges.push_back(range);
            }
        }
    }

    if (state == COUNTING) {
        state = WAITING;
        range.y = height - 1;
        ranges.push_back(range);
    }

    vertHisto.clear();
    int rangSize = ranges.size();

    if (rangSize > 0) {
        std::vector<int> dists;
        dists.push_back(ranges[0].x - 1);

        if (rangSize > 1)
            for (int i = 1; i < rangSize; i++) {
                dists.push_back(ranges[i].x - ranges[i - 1].y - 1);
            }

        dists.push_back(height - ranges[rangSize - 1].y - 1);
        int maxDist = dists[0];
        int maxId = 0;

        for (uint i = 1; i < dists.size(); i++) {
            if (dists[i] >= maxDist) {
                maxDist = dists[i];
                maxId = i;
            }
        }

        if (maxId == 0) {
            hroi.height = maxDist;
        } else {
            hroi.y = ranges[maxId - 1].y + 1;
            hroi.height = maxDist;
        }
    }
    hroi.x = hroi.x < 0 ? 0 : hroi.x;
    hroi.y = hroi.y < 0 ? 0 : hroi.y;
    hroi.width = hroi.x + hroi.width >= width ? width - hroi.x - 1 : hroi.width;
    hroi.height = hroi.y + hroi.height >= height ? height - hroi.y - 1 : hroi.height;
//	std::cout << hroi << " from " << input.size() << std::endl;

	if(hroi.width < 1 || hroi.height < 1)
		output = input.clone();
	else
	{
//		assert(!input(hroi).empty());
		input(hroi).copyTo(output);
	}
}

void vertPrune(const cv::Mat input, cv::Mat &output, int plateType)
{
    int width = input.cols;
    int height = input.rows;
    int counter;
    std::vector<int> histHisto;

    for(int i = 0; i < width; i++)
    {
        counter = 0;

        for (int j = 0; j < height; j++) {
            if (input.at<uchar>(j, i) == 0)
                counter++;
        }

        histHisto.push_back(counter);
    }
    if(histHisto.size() < 30)
    {
        return;
    }
    int delta, upThres, downThres;
    gState state = WAITING;
    cv::Rect roi(0, 0, width, height);

    if(plateType == 1)
    {
        // For Short plate
        delta = 10 < (int)(width - 1.35 * height) ? (int)(width - 1.35 * height) : 10;
        upThres = std::floor(0.8 * height);
        downThres = std::ceil(0.1 * height);

        int stop = 0;
        for(int i = 0; i < delta; i++)
        {
            if(state == WAITING)
            {
                if(histHisto[i] > upThres)
                {
                    state = UP;
                    stop = i + 1;
                }
            }
            else if(state == UP)
            {
                if(histHisto[i] >= upThres)
                {
                    stop = i + 1;
                }
                if(histHisto[i] < downThres)
                {
                    stop = i;
                    state = DOWN;
                }
            }
            else if(state == DOWN)
            {
                if(histHisto[i] >= downThres)
                {
                    stop = i;
                    break;
                }
                else
                    stop = i + 1;
            }
        }
        roi.x = stop;
        if(delta > 15)
            delta = 10 < (int)(width - 1.35 * height - stop) ? (int)(width - 1.35 * height - stop) : 10;
        stop = width - 1;
        state = WAITING;

        for(int i = width - 1; i >= width - delta; i--)
        {
            if(state == WAITING)
            {
                if(histHisto[i] > upThres)
                {
                    state = UP;
                    stop = i - 1;
                }
            }
            else if(state == UP)
            {
                if(histHisto[i] >= upThres)
                {
                    stop = i - 1;
                }
                if(histHisto[i] < downThres)
                {
                    stop = i;
                    state = DOWN;
                }
            }
            else if(state == DOWN)
            {
                if(histHisto[i] >= downThres)
                {
                    stop = i;
                    break;
                }
                else
                    stop = i - 1;
            }
        }
        roi.width = stop - roi.x + 1;
        roi.x = roi.x < 0 ? 0 : roi.x;
        roi.y = roi.y < 0 ? 0 : roi.y;
        roi.width = roi.x + roi.width >= width ? width - roi.x - 1 : roi.width;
        roi.height = roi.y + roi.height >= height ? height - roi.y - 1 : roi.height;
//		std::cout << roi << " from " << input.size() << std::endl;
		if(roi.width < 1 || roi.height < 1)
			output = input.clone();
		else
		{
//			assert(!input(roi).empty());
			input(roi).copyTo(output);
		}
    }
    else
    {
        // For Long Plate
        delta = std::round(0.22 * width);
        int d = std::round(0.5 * height);
        upThres = std::floor(0.9 * height);
        downThres = std::ceil(0.1 * height);

        int stop = 0;
        // Left first prune
        for(int i = 0; i < delta; i++)
        {
            if(state == WAITING)
            {
                if(histHisto[i] > upThres)
                {
                    state = UP;
                    stop = i + 1;
                }
            }
            else if(state == UP)
            {
                if(histHisto[i] >= upThres)
                {
                    stop = i + 1;
                }
                if(histHisto[i] <= 2 * downThres)
                {
                    stop = i;
                    if(histHisto[i] <= downThres)
                    {
                        state = DOWN;
                        break;
                    }
                }
            }
        }
        roi.x = stop;
        state = WAITING;
        // Second left prune
        for(int i = roi.x; i < d + roi.x; i++)
        {
            if(state == WAITING)
            {
                if(histHisto[i] > upThres)
                {
                    state = UP;
                    stop = i + 1;
                }
            }
            else if(state == UP)
            {
				if(histHisto[i] >= upThres)
                {
                    stop = i + 1;
                }
                if(histHisto[i] <= 2 * downThres)
                {
                    stop = i;
                    if(histHisto[i] <= downThres)
                    {
                        state = DOWN;
                        break;
                    }
                }
            }
        }
        roi.x = stop;
        stop = width - 1;
        state = WAITING;
        // Right first prune
        for(int i = width - 1; i >= width - delta; i--)
        {
            if(state == WAITING)
            {
                if(histHisto[i] > upThres)
                {
                    state = UP;
                    stop = i - 1;
                }
            }
            else if(state == UP)
            {
                if(histHisto[i] >= upThres)
                {
                    stop = i - 1;
                }
                if(histHisto[i] <= downThres)
                {
                    stop = i;
                    if(histHisto[i] <= downThres)
                    {
                        state = DOWN;
                        break;
                    }
                }
            }
        }
        int t = stop;
        state = WAITING;
        // Right Second Prune
        for(int i = t; i >= t - d; i--)
        {
            if(state == WAITING)
            {
                if(histHisto[i] > upThres)
                {
                    state = UP;
                    stop = i - 1;
                }
            }
            else if(state == UP)
            {
                if(histHisto[i] >= upThres)
                {
                    stop = i - 1;
                }
                if(histHisto[i] <= downThres)
                {
                    stop = i;
                    if(histHisto[i] <= downThres)
                    {
                        state = DOWN;
                        break;
                    }
                }
            }
        }

        roi.width = stop - roi.x + 1;
        roi.x = roi.x < 0 ? 0 : roi.x;
        roi.y = roi.y < 0 ? 0 : roi.y;
        roi.width = roi.x + roi.width >= width ? width - roi.x - 1 : roi.width;
        roi.height = roi.y + roi.height >= height ? height - roi.y - 1 : roi.height;
//		std::cout << roi << " from " << input.size() << std::endl;
		if(roi.width < 1 || roi.height < 1)
			output = input.clone();
		else
		{
//			assert(!input(roi).empty());
			input(roi).copyTo(output);
		}
    }
}
