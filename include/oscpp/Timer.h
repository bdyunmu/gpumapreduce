#ifndef __OSCPP_TIMER_H__
#define __OSCPP_TIMER_H__

namespace oscpp
{
  /**
   * The timer class gives one access to 10-microsecond accurate timing, and
   * even accuracy is capable if the underlying operating system supports it.
   */
  class Timer
  {
    protected:
      /// A native handle to the "start" time.
      void * t0;
      /// A native handle to the "stop" time.
      void * t1;
      /// A flag indicating whether or not the timer is current running.
      bool running;
    public:
      /// The default constructor which initializes, but does not start, the
      /// timer.
      Timer();
      /// Deallocates the handles to the native timers.
      ~Timer();

      /// @return Whether or not this timer is currently running.
      inline bool isRunning() const { return running; }
      /// Starts or restarts the timer.
      void start();
      /// Stops the timer.
      void stop();

      /**
       * @return The elapsed microseconds between the start and stop of the
       *         timer. If the timer is not stopped when this is called,
       *         returns the elapsed time between the start of the timer and
       *         the current timer.
       */
      double getElapsedMicroseconds() const;

      /**
       * @return The elapsed milliseconds between the start and stop of the
       *         timer. If the timer is not stopped when this is called,
       *         returns the elapsed time between the start of the timer and
       *         the current timer.
       */
      double getElapsedMilliseconds() const;

      /**
       * @return The elapsed seconds between the start and stop of the
       *         timer. If the timer is not stopped when this is called,
       *         returns the elapsed time between the start of the timer and
       *         the current timer.
       */
      double getElapsedSeconds() const;
  };
}

#endif
