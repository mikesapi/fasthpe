#ifndef MA_API_LOCK_H
#define MA_API_LOCK_H

#include "mutex.h"

namespace ftn
{
    namespace mikesapi
    {
        namespace samplecode
        {
            // A very simple scoped-lock class for sample code purposes. 
            // It is recommended that you use the boost threads library.
            class Lock
            {
            public:
                Lock(const Mutex &mutex): _mutex(mutex)
                {
                    _mutex.lock();
                }
                ~Lock()
                {
                    _mutex.unlock();
                }
            private:
                // Noncopyable
                Lock(const Lock &);
                Lock &operator=(const Lock &);
            private:
                const Mutex &_mutex;
            };
        }
    }
}
#endif
