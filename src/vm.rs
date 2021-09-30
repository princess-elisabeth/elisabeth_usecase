use elisabeth::PublicKey;
use fhe_graph::FheGraphExecutor;
use graph_executor::GraphExecutor;
use graph_loader::load::{load, LoadedGraph};
use ndarray::ArrayD;
use std::{collections::HashMap, path::Path, sync::Arc};
use tensor_operator::graph::CpuGraph;
use to::ToVM;

mod to;

pub struct HomomorphicNeuralNetwork(FheGraphExecutor<u64, u64, CpuGraph<u64, u64>>);

impl HomomorphicNeuralNetwork {
    pub fn new(path: &Path) -> Self {
        let LoadedGraph {
            graph_description,
            client_parameters,
            expected_outputs: _,
        } = load::<u64, u64>(path).unwrap();
        let graph = CpuGraph::new(GraphExecutor::new(graph_description).unwrap());
        Self(FheGraphExecutor::new(graph, client_parameters.clone()))
    }

    pub fn run(&self, inputs: HashMap<String, ArrayD<u64>>, pk: &PublicKey) -> Vec<ArrayD<u64>> {
        self.0
            .execute_all_sequential(inputs, HashMap::new(), Arc::new(pk.to_vm()), None, None)
            .unwrap()
    }
}
