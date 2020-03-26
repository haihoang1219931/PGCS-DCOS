#include "UCEventListener.hpp"

UCEventListener* UCEventListener::inst = nullptr;

UCEventListener* UCEventListener::instance() {
    if( inst == nullptr ) {
        inst = new UCEventListener();
    }
    return inst;
}

void UCEventListener::invalidOpenPcdVideo(int invalidCase) {
    Q_EMIT invalidOpenPcdVideoFired(invalidCase);
}

void UCEventListener::pointToPcdFromSidebar(QString pcdUid, bool activeStatus) {
    Q_EMIT userIsPointed(pcdUid, activeStatus);
}
