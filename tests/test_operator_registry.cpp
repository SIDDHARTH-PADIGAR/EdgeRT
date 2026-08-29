#include "edgert/operator_registry.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

#include "edgert/operator.h"
#include "edgert/tensor.h"

using edgert::Operator;
using edgert::OperatorRegistry;
using edgert::Tensor;

namespace {

class EchoOp : public Operator {
public:
    std::string name() const override { return "Echo"; }
    Tensor compute(const std::vector<const Tensor*>& inputs) const override {
        return Tensor(inputs.at(0)->shape(), inputs.at(0)->dtype());
    }
};

}  // namespace

TEST(OperatorRegistryTest, RegisterAndCreate) {
    OperatorRegistry registry;
    registry.register_op("Echo", [] { return std::make_unique<EchoOp>(); });

    EXPECT_TRUE(registry.contains("Echo"));
    EXPECT_FALSE(registry.contains("Missing"));

    auto op = registry.create("Echo");
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->name(), "Echo");
}

TEST(OperatorRegistryTest, CreateUnknownThrows) {
    OperatorRegistry registry;
    EXPECT_THROW(registry.create("DoesNotExist"), std::invalid_argument);
}

TEST(OperatorRegistryTest, DuplicateRegistrationThrows) {
    OperatorRegistry registry;
    registry.register_op("Echo", [] { return std::make_unique<EchoOp>(); });
    EXPECT_THROW(registry.register_op("Echo", [] { return std::make_unique<EchoOp>(); }),
                 std::invalid_argument);
}