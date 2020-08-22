#ifndef IPSEARCH_STATS_H
#define IPSEARCH_STATS_H

namespace ip {

namespace objsearch {

    class Stats
    {
    public:
        int matches;
        int inliers;
        double ratio;
        int keypoints;

        Stats(): matches(0),
            inliers(0),
            ratio(0),
            keypoints(0) {}

        Stats& operator += (const Stats& op)
        {
            matches += op.matches;
            inliers += op.inliers;
            ratio += op.ratio;
            keypoints += op.keypoints;

            return *this;
        }

        Stats& operator /= (int num)
        {
            matches /= num;
            inliers /= num;
            ratio /= num;
            keypoints /= num;

            return *this;
        }
    };

}

}

#endif // IPSEARCH_STATS_H
