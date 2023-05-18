#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "ThreadSafeStack.h"

TEST(Test, PushAndPop) {
  ThreadSafeStack<int> s;

  s.push(1);
  s.push(2);

  EXPECT_EQ(2, s.pop());
  EXPECT_EQ(1, s.pop());
  EXPECT_TRUE(s.empty());
}

TEST(Test, Peek) {
  ThreadSafeStack<int> s;

  s.push(1);
  s.push(2);

  EXPECT_EQ(2, s.peek());
  EXPECT_EQ(2, s.peek());
  EXPECT_EQ(2, s.pop());
  EXPECT_EQ(1, s.peek());
}

TEST(Test, ConcurrentPushAndPop) {
  ThreadSafeStack<int> s;
  std::vector<std::thread> threads;

  for (int i = 0; i < 1000; i++) {
    threads.emplace_back([&s, i]() {
      s.push(i);
      s.pop();
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_TRUE(s.empty());
}
TEST(Test, MoreConcurrentPushAndPop) {
  ThreadSafeStack<int> s;
  std::vector<std::thread> threads;

  for (int i = 0; i < 100; i++) {
    threads.push_back(std::thread([&s, i]() {
      for (int j = 0; j < 100; j++) {
        s.push(i * 100 + j);
      }
    }));
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_FALSE(s.empty());

  std::vector<int> results(100 * 100);
  std::vector<std::thread> popThreads;

  for (int i = 0; i < 100; i++) {
    popThreads.push_back(std::thread([&s, &results, i]() {
      for (int j = 0; j < 100; j++) {
        results[i * 100 + j] = s.pop();
      }
    }));
  }

  for (auto& t : popThreads) {
    t.join();
  }

  std::sort(results.begin(), results.end());

  for (int i = 0; i < 100 * 100; i++) {
    EXPECT_EQ(results[i], i);
  }

  EXPECT_TRUE(s.empty());
}
TEST(Test, ConcurrentPeek) {
  ThreadSafeStack<int> s;
  s.push(1);
  s.push(2);

  std::vector<std::thread> threads;

  for (int i = 0; i < 1000; i++) {
    threads.emplace_back([&s]() { EXPECT_TRUE(s.peek() == 2); });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(2, s.pop());
  EXPECT_EQ(1, s.pop());
  EXPECT_TRUE(s.empty());
}

TEST(Test, WaitForPush) {
  ThreadSafeStack<int> s;
  int val = 0;
  std::thread t1([&s, &val]() {
    s.peek();
    val = s.pop();
  });
  std::thread t2([&s]() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    s.push(1);
  });
  t2.join();
  t1.join();

  EXPECT_EQ(val, 1);
  EXPECT_TRUE(s.empty());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
