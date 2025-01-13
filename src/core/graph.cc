#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/transpose.h"
#include "operators/matmul.h"

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    bool isInverse(const TransposeObj &self, const TransposeObj &other) {
        auto self_permute = self.getPermute();
        auto other_permute = other.getPermute();
        for (int i = 0; i < (int)self_permute.size(); ++i) {
            if (self_permute[other_permute[i]] != i) {
                return false;
            }
        }
        return true;
    }

    int isTransMat(const TransposeObj &self) {
        auto permute = self.getPermute();
        auto rank = permute.size();
        if (rank <= 1) {
            return -1;
        }
        for (int i = 0; i < (int)rank - 2; ++i) {
            if (permute[i] != i) {
                return -1;
            }
        }
        return permute[rank - 1] == (int)rank - 1 ? 0 : 1;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        IT_ASSERT(topo_sort());
        auto remove_ops = std::vector<Operator>();
        auto remove_tensors = std::vector<Tensor>();
        auto wait_for_cut = std::vector<Tensor>();
        for (auto const &op : ops) {
            if (op->getOpType() == OpType::Transpose && op->getPredecessors().size() == 1) {

                auto last_op = op->getPredecessors()[0];
                if (last_op->getOpType() == OpType::Transpose) {
                    if (!isInverse(*(static_cast<TransposeObj*>(op.get())), *(static_cast<TransposeObj*>(last_op.get())))) {
                        continue;
                    }
                    auto last_data = last_op->getInputs()[0];
                    // 去除 op 与其他算子的连接
                    last_op->removeSuccessors(op);
                    op->removePredecessors(last_op);
                    for (auto const &nxt_op : op->getSuccessors()) {
                        op->removeSuccessors(nxt_op);
                        nxt_op->removePredecessors(op);
                    }
                    // 删除 op
                    remove_ops.push_back(op);
                    // 如果 last_last_op 存在，则将所有的 nxt_op 接到 last_last_op 上
                    if (last_op->getPredecessors().size() == 1) {
                        auto last_last_op = last_op->getPredecessors()[0];
                        for (auto const &nxt_op : op->getSuccessors()) {
                            nxt_op->addPredecessors(last_last_op);
                            last_last_op->addSuccessors(nxt_op);
                        }
                    }
                    // 删除 op 的 output（用 last_data 替换）
                    // 修改 Source 和 Target
                    for (auto const &input : op->getInputs()) {
                        input->removeTarget(op);
                        if (input->getTargets().size() == 0) {
                            wait_for_cut.push_back(input);
                        }
                    }
                    for (auto const &output : op->getOutputs()) {
                        for (auto const &target : output->getTargets()) {
                            last_data->addTarget(target);
                            target->replaceInput(output, last_data);
                        }
                        remove_tensors.push_back(output);
                    }
                }
            }
        }
        for (auto const &op : ops) {
            if (op->getOpType() == OpType::MatMul) {
                IT_ASSERT(op->getInputs().size() == 2);
                auto data_1 = op->getInputs()[0];
                if (auto last_op_1 = data_1->getSource()) {
                    if (last_op_1->getOpType() == OpType::Transpose) {
                        auto trans_mat_1 = isTransMat(*(static_cast<TransposeObj*>(last_op_1.get())));
                        if (trans_mat_1 >= 0) {
                            // ? -> last_data_1 -> last_op_1 -> data_1 -> op -> output
                            // ? -> last_data_1 -> op -> output
                            // 修改 last_op_1 附近算子的连接
                            // 注意这里 last_op_1 和 data_1 不能直接删掉，因为它可能被其他算子复用
                            last_op_1->removeSuccessors(op);
                            op->removePredecessors(last_op_1);
                            if (last_op_1->getPredecessors().size() == 1) {
                                auto last_last_op_1 = last_op_1->getPredecessors()[0];
                                last_last_op_1->removeSuccessors(last_op_1);
                                last_op_1->removePredecessors(last_last_op_1);
                                last_op_1->addSuccessors(op);
                                op->addPredecessors(last_op_1);
                            }
                            // 修改 last_data_1, data_1 的 Source 和 Target
                            auto last_data_1 = last_op_1->getInputs()[0];
                            if (data_1->getTargets().size() == 0) {
                                wait_for_cut.push_back(data_1);
                            }
                            op->replaceInput(data_1, last_data_1);
                            last_data_1->addTarget(op);
                            // 修改 op 的属性
                            auto matmul_op = static_cast<MatmulObj*>(op.get());
                            matmul_op->setTransA(matmul_op->getTransA() != bool(trans_mat_1));
                        }
                    }
                }
                auto data_2 = op->getInputs()[1];
                if (auto last_op_2 = data_2->getSource()) {
                    if (last_op_2->getOpType() == OpType::Transpose) {
                        auto trans_mat_2 = isTransMat(*(static_cast<TransposeObj*>(last_op_2.get())));
                        if (trans_mat_2 >= 0) {
                            // ? -> last_data_2 -> last_op_2 -> data_2 -> op -> output
                            // ? -> last_data_2 -> op -> output
                            // 修改 last_op_2 附近算子的连接
                            // 注意这里 last_op_2 和 data_2 不能直接删掉，因为它可能被其他算子复用
                            last_op_2->removeSuccessors(op);
                            op->removePredecessors(last_op_2);
                            if (last_op_2->getPredecessors().size() == 1) {
                                auto last_last_op_2 = last_op_2->getPredecessors()[0];
                                last_last_op_2->removeSuccessors(last_op_2);
                                last_op_2->removePredecessors(last_last_op_2);
                                last_op_2->addSuccessors(op);
                                op->addPredecessors(last_op_2);
                            }
                            // 修改 last_data_2, data_2 的 Source 和 Target
                            auto last_data_2 = last_op_2->getInputs()[0];
                            data_2->removeTarget(op);
                            if (data_2->getTargets().size() == 0) {
                                wait_for_cut.push_back(data_2);
                            }
                            op->replaceInput(data_2, last_data_2);
                            last_data_2->addTarget(op);
                            // 修改 op 的属性
                            auto matmul_op = static_cast<MatmulObj*>(op.get());
                            matmul_op->setTransB(matmul_op->getTransB() != bool(trans_mat_2));
                        }
                    }
                }
            }
        }
        while (wait_for_cut.size() > 0) {
            auto tensor = wait_for_cut.back();
            wait_for_cut.pop_back();
            if (auto op = tensor->getSource()) {
                // 删除 op 的所有连接
                for (auto const &pred: op->getPredecessors()) {
                    pred->removeSuccessors(op);
                    op->removePredecessors(pred);
                }
                // 删除 op 与它的所有 Input 的连接
                for (auto const &input: op->getInputs()) {
                    input->removeTarget(op);
                    if (input->getTargets().size() == 0) {
                        wait_for_cut.push_back(input);
                    }
                }
                // 删除 op
                remove_ops.push_back(op);
                remove_tensors.push_back(tensor);
            }
        }
        for (auto const &op: remove_ops) {
            ops.erase(std::remove(ops.begin(), ops.end(), op), ops.end());
        }
        for (auto const &tensor: remove_tensors) {
            tensors.erase(std::remove(tensors.begin(), tensors.end(), tensor), tensors.end());
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        size_t total_size = 0;
        for (auto &tensor : tensors) {
            total_size += tensor->size() * tensor->getDType().getSize();
        }
        size_t addr = allocator.alloc(total_size);
        for (auto &tensor : tensors) {
            auto ptr = static_cast<char*>(allocator.getPtr()) + addr;
            tensor->setDataBlob(make_ref<BlobObj>(runtime, ptr));
            addr += tensor->size() * tensor->getDType().getSize();
        }
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini