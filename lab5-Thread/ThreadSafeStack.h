#include <condition_variable>
#include <mutex>
#include <stack>

template <typename T>
class ThreadSafeStack {
 public:
  T peek() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return !data_.empty(); });
    return data_.top();
  }

  void push(T value) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      data_.push(value);
    }
    cv_.notify_all();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return !data_.empty(); });
    T top = data_.top();
    data_.pop();
    return top;
  }

  bool empty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return data_.empty();
  }

 private:
  std::condition_variable cv_;
  std::stack<T> data_;
  std::mutex mutex_;
};
