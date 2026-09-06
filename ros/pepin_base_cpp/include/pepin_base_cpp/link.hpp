// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// A reconnecting JSON-lines TCP client — the C++ twin of pepin_bringup/link.py.
//
// The board's servers publish forever and go away without warning (a service restart,
// a wedged bus, a reboot). A ROS node must survive that without dying and without ever
// blocking its executor, so the socket lives here on its own thread: connect, read,
// decode, hand each message to a callback; on any failure sleep a growing backoff and
// try again. Nothing in this header includes ROS.
//
// Ownership of the two threads: the reader thread calls `on_message`, so whatever the
// node does there must be thread-safe. `send` is called from the ROS thread and never
// blocks for long on a dead link — the socket carries a 0.2 s send timeout, and a
// failed write drops the line and says so.

#ifndef PEPIN_BASE_CPP__LINK_HPP_
#define PEPIN_BASE_CPP__LINK_HPP_

#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "pepin_base_cpp/protocol.hpp"

namespace pepin
{

/// A background TCP link to a JSON-lines server: messages out to a callback, lines in.
class JsonLineLink
{
public:
  using MessageHandler = std::function<void (const nlohmann::json &)>;

  /// Prepare a link to `host:port` called `name` in logs; nothing connects until start().
  JsonLineLink(
    std::string host, int port, MessageHandler on_message, std::string name,
    double min_backoff_s = 0.5, double max_backoff_s = 5.0)
  : host_(std::move(host)),
    port_(port),
    on_message_(std::move(on_message)),
    name_(std::move(name)),
    min_backoff_s_(min_backoff_s),
    max_backoff_s_(max_backoff_s) {}

  JsonLineLink(const JsonLineLink &) = delete;
  JsonLineLink & operator=(const JsonLineLink &) = delete;

  /// Stop the reader thread and close the socket.
  ~JsonLineLink() {stop();}

  /// Start the reader thread; it connects, and reconnects, on its own.
  void start()
  {
    if (thread_.joinable()) {
      return;
    }
    thread_ = std::thread(&JsonLineLink::run, this);
  }

  /// Ask the reader thread to finish and close the socket; safe to call twice.
  void stop()
  {
    {
      const std::lock_guard<std::mutex> guard(stop_mutex_);
      stop_ = true;
    }
    stop_cv_.notify_all();
    {
      const std::lock_guard<std::mutex> guard(mutex_);
      if (fd_ >= 0) {
        ::shutdown(fd_, SHUT_RDWR);  // a blocking recv returns at once
      }
    }
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  /// Write one line to the server; false means the link was down and it was dropped.
  bool send(const std::string & line)
  {
    // The lock is held across the write on purpose: it keeps the reader thread from
    // closing (and the kernel from reusing) this descriptor mid-send. Bounded by the
    // socket's 0.2 s send timeout, so the ROS executor cannot hang here.
    const std::lock_guard<std::mutex> guard(mutex_);
    if (fd_ < 0) {
      return false;
    }
    std::size_t sent = 0;
    while (sent < line.size()) {
      const ssize_t written = ::send(fd_, line.data() + sent, line.size() - sent, kNoSignal);
      if (written <= 0) {
        if (written < 0 && errno == EINTR) {
          continue;
        }
        return false;
      }
      sent += static_cast<std::size_t>(written);
    }
    return true;
  }

  /// The (connected, detail) the link last changed to, once, or nothing if unchanged.
  ///
  /// Repeated failures of the same kind report once, so a node can log every call and
  /// still not flood the console while the server is down.
  std::optional<std::pair<bool, std::string>> take_status_change()
  {
    const std::lock_guard<std::mutex> guard(mutex_);
    std::optional<std::pair<bool, std::string>> change;
    change.swap(status_change_);
    return change;
  }

  /// Whether a socket to the server is open right now.
  bool connected() const
  {
    const std::lock_guard<std::mutex> guard(mutex_);
    return connected_;
  }

private:
  static constexpr std::size_t kRecvBytes = 4096;
  static constexpr int kRecvTimeoutMs = 200;   // how often the reader checks for a stop
  static constexpr double kConnectTimeoutS = 2.0;
#ifdef MSG_NOSIGNAL
  static constexpr int kNoSignal = MSG_NOSIGNAL;  // a dead peer must not raise SIGPIPE
#else
  static constexpr int kNoSignal = 0;
#endif

  /// Reader thread: connect, read until the link breaks, back off, repeat.
  void run()
  {
    double backoff_s = min_backoff_s_;
    while (!stopped()) {
      std::string error;
      const int fd = connect_to_server(error);
      if (fd < 0) {
        set_status(false, name_ + " unreachable at " + where() + ": " + error);
        wait_for_stop(backoff_s);
        backoff_s = std::min(backoff_s * 2.0, max_backoff_s_);
        continue;
      }
      backoff_s = min_backoff_s_;
      set_status(true, name_ + " connected at " + where());
      read_until_closed(fd);
    }
    set_status(false, name_ + " link closed");
  }

  /// Feed every line of one connection to the callback until it breaks or we stop.
  void read_until_closed(int fd)
  {
    LineReader reader;
    {
      const std::lock_guard<std::mutex> guard(mutex_);
      fd_ = fd;
    }
    char buffer[kRecvBytes];
    while (!stopped()) {
      const ssize_t received = ::recv(fd, buffer, sizeof(buffer), 0);
      if (received < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
          continue;  // the recv timeout: just a chance to notice a stop
        }
        set_status(false, name_ + " read failed: " + std::strerror(errno));
        break;
      }
      if (received == 0) {
        set_status(false, name_ + " closed the connection");
        break;
      }
      for (const auto & message : reader.feed(buffer, static_cast<std::size_t>(received))) {
        on_message_(message);
      }
    }
    const std::lock_guard<std::mutex> guard(mutex_);
    fd_ = -1;
    connected_ = false;
    ::close(fd);
  }

  /// A TCP socket to the server with a 2 s connect timeout; -1 and `error` when it fails.
  int connect_to_server(std::string & error)
  {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo * results = nullptr;
    const std::string service = std::to_string(port_);
    const int status = ::getaddrinfo(host_.c_str(), service.c_str(), &hints, &results);
    if (status != 0) {
      error = ::gai_strerror(status);
      return -1;
    }
    int fd = -1;
    error = "no address";
    for (const addrinfo * it = results; it != nullptr && fd < 0; it = it->ai_next) {
      fd = connect_one(*it, error);
    }
    ::freeaddrinfo(results);
    return fd;
  }

  /// Connect to one resolved address, non-blocking with a deadline, then set the timeouts.
  int connect_one(const addrinfo & address, std::string & error)
  {
    const int fd = ::socket(address.ai_family, address.ai_socktype, address.ai_protocol);
    if (fd < 0) {
      error = std::strerror(errno);
      return -1;
    }
    const int flags = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    if (::connect(fd, address.ai_addr, address.ai_addrlen) != 0) {
      if (errno != EINPROGRESS) {
        error = std::strerror(errno);
        ::close(fd);
        return -1;
      }
      if (!wait_writable(fd, error)) {  // still handshaking: give it the connect timeout
        ::close(fd);
        return -1;
      }
    }
    ::fcntl(fd, F_SETFL, flags);
    timeval timeout{0, kRecvTimeoutMs * 1000};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    return fd;
  }

  /// Wait out an in-flight connect; false (with `error`) on timeout or refusal.
  bool wait_writable(int fd, std::string & error)
  {
    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration<double>(kConnectTimeoutS);
    for (;;) {
      const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now()).count();
      if (left <= 0) {
        error = "timed out";
        return false;
      }
      pollfd waiting{fd, POLLOUT, 0};
      const int ready = ::poll(&waiting, 1, static_cast<int>(left));
      if (ready < 0) {
        if (errno == EINTR) {
          continue;
        }
        error = std::strerror(errno);
        return false;
      }
      if (ready == 0) {
        error = "timed out";
        return false;
      }
      int failure = 0;
      socklen_t size = sizeof(failure);
      if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &failure, &size) != 0 || failure != 0) {
        error = std::strerror(failure != 0 ? failure : errno);
        return false;
      }
      return true;
    }
  }

  /// Record the link's state; keep the first `detail` of each change for the node to log.
  void set_status(bool connected, const std::string & detail)
  {
    const std::lock_guard<std::mutex> guard(mutex_);
    connected_ = connected;
    if (!reported_.has_value() || *reported_ != connected) {
      reported_ = connected;
      status_change_ = std::make_pair(connected, detail);
    }
  }

  /// Whether stop() has been called.
  bool stopped()
  {
    const std::lock_guard<std::mutex> guard(stop_mutex_);
    return stop_;
  }

  /// Sleep for the backoff, or until stop() cuts it short.
  void wait_for_stop(double seconds)
  {
    std::unique_lock<std::mutex> guard(stop_mutex_);
    stop_cv_.wait_for(guard, std::chrono::duration<double>(seconds), [this] {return stop_;});
  }

  /// The address as host:port, for log lines.
  std::string where() const {return host_ + ":" + std::to_string(port_);}

  std::string host_;
  int port_;
  MessageHandler on_message_;
  std::string name_;
  double min_backoff_s_;
  double max_backoff_s_;

  mutable std::mutex mutex_;  // guards the descriptor and the status below
  int fd_ = -1;
  bool connected_ = false;
  std::optional<bool> reported_;
  std::optional<std::pair<bool, std::string>> status_change_;

  std::mutex stop_mutex_;
  std::condition_variable stop_cv_;
  bool stop_ = false;
  std::thread thread_;
};

}  // namespace pepin

#endif  // PEPIN_BASE_CPP__LINK_HPP_
