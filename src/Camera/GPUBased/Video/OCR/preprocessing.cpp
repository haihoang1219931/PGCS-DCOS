#include "preprocessing.h"

//===============================
// I. Main Functions            "
//===============================
std::vector<cv::Mat> preprocess(cv::Mat grayImg, int plateType, int *sign)
{
    cv::Mat binImg;
    if((grayImg.rows > 26 && plateType == 0) || (grayImg.rows > 54 && plateType == 1))
            binarize(grayImg, binImg, 9);
        else
            binarize(grayImg, binImg, 5);
    grayImg.release();
    cv::Mat filImg;
    filter(binImg, filImg);
    cv::imwrite("/home/pgcs-04/workspace/giapvn/gray/" + std::to_string(00) + ".png", filImg, {CV_IMWRITE_PNG_COMPRESSION, 0});


    binImg.release();
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
        vertPrune(temp, prunedImg, plateType);
        chars = extractCharsSP(prunedImg, sign);
    }
    filImg.release();
    prunedImg.release();
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

            if (rect.height > 12 && rect.width > 3 && rect.width < 0.5 * width && rect.height > 0.25 * height && rect.height < 0.5 * height)
            {
                rects.push_back(rect);
//                cv::rectangle(input, rect, 127);
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
                if(std::abs(fRects[i].y - sRects[sStd].y) > (int)(0.7 * sRects[sStd].height))
                    firstLine.push_back(fRects[i]);
            }
            for(uint i = 0; i < sRects.size(); i++)
            {
                if(std::abs(sRects[i].y - fRects[fStd].y) > (int)(0.7 * fRects[fStd].height))
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
        std::vector<cv::Rect> fLine, sLine;
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
                    fLine.push_back(r);
                }
            }
            else
                fLine.push_back(firstLine[i]);
        }
        for(uint i = 0; i < secondLine.size(); i++)
        {
            float ratio = (float)secondLine[i].height / (float)secondLine[i].width;
            if(hpw / ratio > 1.9)
            {
                // Too long
                int numSeg = std::round(hpw / ratio / 1.15);
                int w = std::ceil(secondLine[i].width / numSeg);
                for(int j = 0; j < numSeg; j++)
                {
                    cv::Rect r(secondLine[i].x + w * j, secondLine[i].y, w + 1, secondLine[i].height);
                    sLine.push_back(r);
                }
            }
            else
                sLine.push_back(secondLine[i]);
        }
        firstLine.clear();
        secondLine.clear();
        // End 4 ./.
        // 5. Expand 2 sides
        int sLineSize = sLine.size();
        int w = std::round(stdHeight / hpw);
        int dw = std::round(1.4 * stdHeight / hpw);

        if(sLine.size() < 5)
        {
            int numLack = 5 - sLineSize;
            cv::Rect mark = sLine[0];
            for(int i = 0; i < numLack; i++)
            {
                cv::Rect r(0, mark.y, 0, mark.height);
                if(mark.x - 0.8 * w >= 0)
                {
                    r.x = mark.x - dw < 0 ? 0 : mark.x - dw;
                    r.width = mark.x - r.x - (int)(0.25 * w);
                    sLine.push_back(r);
                    mark = r;
                }
                else
                    break;
            }
            mark = sLine[sLineSize - 1];
            for(int i = 0; i < numLack; i++)
            {
                cv::Rect r(0, mark.y, 0, mark.height);
                if(mark.x + mark.width + 0.8 * w < width)
                {
                    r.x = mark.x + mark.width + (int)(0.25 * w) - 1;
                    r.width = r.x + (int)(1.15 * w) > width ? width - r.x : (int)(1.15 * w);
                    sLine.push_back(r);
                    mark = r;
                }
                else
                    break;
            }
            std::sort(sLine.begin(), sLine.end(), [](cv::Rect a, cv::Rect b)
            {
                return a.x < b.x;
            });
            sLineSize = sLine.size();
        }
        if(fLine[0].x < sLine[0].x - sLine[0].width / 3)
            fLine.erase(fLine.begin());
        if(fLine[fLine.size() - 1].x + fLine[fLine.size() - 1].width > sLine[sLineSize - 1].x + 1.2 * sLine[sLineSize - 1].width)
            fLine.erase(fLine.end() - 1);
        int fLineSize = fLine.size();
        if(fLine.size() < 5)
        {
            cv::Rect mark = fLine[0];
            if(hpw <= 2.1)
            {
                // Is car plate
                while(true)
                {
                    cv::Rect r(0, mark.y, 0, mark.height);
                    r.x = mark.x - dw < 0 ? 0 : mark.x - dw;
                    r.width = mark.x - r.x - (int)(0.25 * w);
                    if(r.x >= sLine[0].x + sLine[0].width / 2)
                    {
                        fLine.push_back(r);
                        mark = r;
                    }
                    else
                    {
                        if(mark.x <= sLine[0].x + sLine[0].width / 2 && fLine.size() > 3)
                        {
                            fLine.erase(fLine.begin());
                        }
                        break;
                    }
                }
                if(fLineSize != fLine.size())
                    std::sort(fLine.begin(), fLine.end(), [](cv::Rect a, cv::Rect b)
                    {
                        return a.x < b.x;
                    });
                mark = fLine[fLine.size() - 1];
                while(true)
                {
                    cv::Rect r(0, mark.y, 0, mark.height);
                    r.x = mark.x + mark.width + (int)(0.25 * w);
                    if(r.x < sLine[sLineSize - 1].x + sLine[sLineSize - 1].width / 3)
                    {
                        r.width = (r.x + (int)(1.1 * w)) < width ? (int)(1.15 * w) : width - r.x + 1;
                        fLine.push_back(r);
                        mark = r;
                    }
                    else
                    {
                        if(mark.x + mark.width >= sLine[sLineSize - 1].x + sLine[sLineSize - 1].width / 2 && fLine.size() > 3)
                        {
                            fLine.erase(fLine.end() - 1);
                        }
                        break;
                    }
                }

            }
            else
            {
                // Is motor plate
                while(true)
                {
                    cv::Rect r(0, mark.y, 0, mark.height);
                    r.x = mark.x - dw < 0 ? 0 : mark.x - dw;
                    r.width = mark.x - r.x - (int)(0.25 * w);
                    if(r.x >= sLine[0].x + sLine[0].width / 3)
                    {
                        fLine.push_back(r);
                        mark = r;
                    }
                    else
                    {
                        if(mark.x <= sLine[0].x - sLine[0].width / 3 && fLine.size() > 5)
                        {
                            fLine.erase(fLine.begin());
                        }
                        break;
                    }
                }
                if(fLineSize != fLine.size())
                    std::sort(fLine.begin(), fLine.end(), [](cv::Rect a, cv::Rect b)
                    {
                        return a.x < b.x;
                    });
                mark = fLine[fLine.size() - 1];
                while(true)
                {
                    cv::Rect r(0, mark.y, 0, mark.height);
                    r.x = mark.x + mark.width + (int)(0.25 * w);
                    if(r.x < sLine[sLineSize - 1].x + sLine[sLineSize - 1].width / 3)
                    {
                        r.width = (r.x + (int)(1.1 * w)) < width ? (int)(1.15 * w) : width - r.x;
                        fLine.push_back(r);
                        mark = r;
                    }
                    else
                    {
                        if(mark.x + mark.width >= sLine[sLineSize - 1].x + 1.2 * sLine[sLineSize - 1].width && fLine.size() > 5)
                        {
                            fLine.erase(fLine.end() - 1);
                        }
                        break;
                    }
                }
            }
        }

        // End 5 ./.
        // Get characters
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
            sLine[i].y -= dh;
            sLine[i].height += dh;
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
    input(hroi).copyTo(output);
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

        input(roi).copyTo(output);
    }
    else
    {
        // For Long Plate
        delta = std::round(0.22 * width);
        int d = std::round(0.5 * height);
        upThres = std::floor(0.9 * height);
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
        input(roi).copyTo(output);
    }
}
