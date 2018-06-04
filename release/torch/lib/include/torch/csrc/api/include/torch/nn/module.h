#pragma once

#include "torch/detail.h"

#include <torch/csrc/autograd/variable.h>

#include <ATen/optional.h>

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace torch {
namespace nn {

class Module {
 public:
  /// Tells the base `Module` about the name of the submodule.
  explicit Module(std::string name);

  /// Constructs the base module without immediate knowledge of the submodule's
  /// name. The name of the submodule is inferred via RTTI the first time
  /// `.name()` is invoked.
  Module() = default;

  virtual ~Module() = default;

  /// Returns the name of the `Module`.
  const std::string& name() const noexcept;

  virtual std::shared_ptr<Module> clone() const;

  // Only construct parameters in initialize_parameters, and
  // containers in initialize_containers. Most of the time, the containers are
  // the only thing you need to add.
  // You are guaranteed that containers are added before parameters.
  virtual void initialize_containers() {}
  virtual void initialize_parameters() {}
  virtual void reset_parameters() {}

  std::map<std::string, Variable> parameters() const;
  Variable& param(std::string const&);

  /// Enables training mode.
  virtual void train();

  /// Disables training mode.
  virtual void eval();

  /// True if the module is in training mode.
  virtual bool is_training() const noexcept;

  /// Recursively moves all parameters to CPU memory (in place).
  virtual void cpu();

  /// Recursively moves all parameters to CUDA memory (in place).
  virtual void cuda();

  /// Recursively casts all parameters to the given type.
  virtual void to(at::Type& type);

  /// Recursively casts all parameters to the given scalar type.
  virtual void to(at::ScalarType scalar_type);

  /// Recursively moves all parameters to the given backend.
  virtual void to(at::Backend backend);

  /// Recursively zeros out the `grad` values of all parameters.
  virtual void zero_grad();

  template <class Archive>
  void save(Archive& ar) const {
    auto params = parameters();
    std::size_t size = params.size();
    ar(size);
    for (auto& p : params) {
      ar(p.first, p.second);
    }
  }

  template <class Archive>
  void load(Archive& ar) {
    auto params = parameters();
    std::size_t size;
    ar(size);
    std::string name;
    for (std::size_t i = 0; i < size; i++) {
      ar(name);
      ar(params[name]);
    }
  }

 protected:
  Variable register_parameter(const std::string& name, at::Tensor tensor);
  Variable register_buffer(const std::string& name, at::Tensor tensor);

  template <typename ModuleType>
  std::shared_ptr<ModuleType> register_module(
      const std::string& name,
      const std::shared_ptr<ModuleType>& module) {
    const auto pair = children_.emplace(name, module);
    AT_CHECK(pair.second, "Module has already been registered");
    return module;
  }

 private:
  template <typename Derived>
  friend class CloneableModule;

  virtual void clone_(Module& other);

  std::unordered_map<std::string, Variable> parameters_;
  std::unordered_map<std::string, std::shared_ptr<Module>> children_;

  /// The module's name (e.g. "LSTM").
  mutable at::optional<std::string> name_;

  /// Whether the module is in training mode.
  bool is_training_{true};
};

/// The `clone()` method in the base `Module` class does not have knowledge of
/// the concrete runtime type of its subclasses. Therefore, `clone()` must
/// either be called from within the subclass, or from a base class that has
/// knowledge of the concrete type. `CloneableModule` uses the CRTP to gain
/// knowledge of the subclass' static type and provide an implementation of the
/// `clone()` method. We do not want to use this pattern in the base class,
/// because then storing a module would always require templatizing it.
template <typename Derived>
class CloneableModule : public Module {
 public:
  using Module::Module;

  virtual void reset() = 0;

  /// Moves the `Module` into a `shared_ptr` and calls `reset()` on it.
  std::shared_ptr<Derived> build() {
    auto module = std::make_shared<Derived>(static_cast<Derived&&>(*this));
    module->reset();
    return std::move(module);
  }

  /// Performs a recursive "deep copy" of the `Module`, such that all parameters
  /// and submodules in the cloned module are different from those in the
  /// original module.
  std::shared_ptr<Module> clone() const override {
    const auto& self = static_cast<const Derived&>(*this);
    auto copy = std::make_shared<Derived>(self);
    copy->parameters_.clear();
    copy->children_.clear();
    copy->reset();
    for (auto& parameter : parameters_) {
      copy->parameters_.at(parameter.first)
          .data()
          .copy_(parameter.second.data());
    }
    for (auto& child : children_) {
      copy->children_.at(child.first)->clone_(*child.second);
    }
    return copy;
  }

 private:
  void clone_(Module& other) final override {
    // Here we are *pretty* certain that `other's` type is `Derived` (because it
    // was registered under the same name as `this`), but you never know what
    // crazy things `reset()` does, so `dynamic_cast` just to be safe.
    auto clone = std::dynamic_pointer_cast<Derived>(other.clone());
    AT_CHECK(
        clone != nullptr,
        "Attempted to clone submodule, but it is of a "
        "different type than the submodule it was to be cloned into");
    static_cast<Derived&>(*this) = std::move(*clone);
  }
};
} // namespace nn
} // namespace torch

#define TORCH_ATTR(T, name)                         \
  auto name(const T& new_##name)->decltype(*this) { \
    this->name##_ = new_##name;                     \
    return *this;                                   \
  }                                                 \
  auto name(T&& new_##name)->decltype(*this) {      \
    this->name##_ = std::move(new_##name);          \
    return *this;                                   \
  }                                                 \
  const T& name() const noexcept {                  \
    return this->name##_;                           \
  }                                                 \
  T name##_
