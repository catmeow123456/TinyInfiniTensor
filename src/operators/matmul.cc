#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto A_shape = inputs[0]->getDims();
        auto B_shape = inputs[1]->getDims();
        auto A_rank = A_shape.size();
        auto B_rank = B_shape.size();
        assert(A_rank >= 2 && B_rank >= 2);
        if (transA) {
            std::swap(A_shape[A_rank - 2], A_shape[A_rank - 1]);
        }
        if (transB) {
            std::swap(B_shape[B_rank - 2], B_shape[B_rank - 1]);
        }
        A_shape[A_rank - 1] = 1;
        B_shape[B_rank - 2] = 1;
        auto output_shape = infer_broadcast(A_shape, B_shape);
        return {{output_shape}};
    }

} // namespace infini