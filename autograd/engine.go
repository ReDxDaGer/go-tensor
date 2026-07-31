package autograd

import tensor "github.com/redxdager/go-tensor/tensors"

func Backward(t *tensor.Tensor) {
	if t.Grad == nil {
		t.Grad = tensor.Ones(t.Shape...)
	} else {
		for i := range t.Grad.Data {
			t.Grad.Data[i] = 1.0
		}
	}

	topo := []*tensor.Tensor{}
	visited := make(map[*tensor.Tensor]bool)
	var buildTopo func(node *tensor.Tensor)
	buildTopo = func(node *tensor.Tensor) {
		if !visited[node] {
			visited[node] = true
			for _, parent := range node.Parents {
				if parent != nil {
					buildTopo(parent)
				}
			}
			topo = append(topo, node)
		}
	}
	buildTopo(t)

	// 2. Execute backward closures in reverse topological order
	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]
		if node.Backward != nil {
			node.Backward()
		}
	}
}
func ZeroGrad(tensorsToReset ...*tensor.Tensor) {
	for _, t := range tensorsToReset {
		if t != nil && t.Grad != nil {
			for i := range t.Grad.Data {
				t.Grad.Data[i] = 0.0
			}
		}
	}
}
