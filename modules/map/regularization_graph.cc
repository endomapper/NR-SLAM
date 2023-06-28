/*
 * This file is part of NR-SLAM
 *
 * Copyright (C) 2022-2023 Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * NR-SLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "regularization_graph.h"

#include "utilities/geometry_toolbox.h"

#include "absl/log/log.h"

using namespace std;

RegularizationGraph::RegularizationGraph(Options& options, Map* map) :
    options_(options), map_(map) {
    min_weight_ = InterpolationWeight(options_.weight_sigma * 1.5, options_.weight_sigma);
}

void RegularizationGraph::SetSigma(const float sigma) {
    options_.weight_sigma = sigma;
    min_weight_ = InterpolationWeight(options_.weight_sigma * 1.5, options_.weight_sigma);
}

void RegularizationGraph::AddEdge(ID mappoint_id, ID mappoint_id_other,
                                  Eigen::Vector3f& relative_position) {
    float distance = relative_position.norm();

    auto edge = make_shared<Edge>();
    edge->vertex_id_1 = mappoint_id;
    edge->vertex_id_2 = mappoint_id_other;
    edge->distance = distance;
    edge->first_distance = distance;
    edge->weight = InterpolationWeight(distance, options_.weight_sigma);
    edge->status = NEUTRAL;
    edge->max_distance = distance;
    edge->min_distance = distance;
    edge->last_relative_position = relative_position;

    graph_[mappoint_id][mappoint_id_other] = edge;
    graph_[mappoint_id_other][mappoint_id] = edge;
}

std::shared_ptr<RegularizationGraph::Edge> RegularizationGraph::GetEdge(ID mappoint_id, ID mappoint_id_other) {
    return graph_[mappoint_id][mappoint_id_other];
}

bool EdgeComparator(pair<ID, std::shared_ptr<RegularizationGraph::Edge>> edge_1,
                    pair<ID, std::shared_ptr<RegularizationGraph::Edge>> edge_2){
    if(edge_1.second->status != edge_2.second->status){
        return edge_1.second->status < edge_2.second->status;
    }
    else{
        return edge_1.second->weight > edge_2.second->weight;
    }
}

std::vector<std::pair<ID, std::shared_ptr<RegularizationGraph::Edge>>>
        RegularizationGraph::GetEdges(ID mappoint_id) const {
    vector<pair<ID, std::shared_ptr<RegularizationGraph::Edge>>> edges(graph_.at(mappoint_id).begin(),
                                                      graph_.at(mappoint_id).end());
    sort(edges.begin(), edges.end(), EdgeComparator);

    std::vector<std::pair<ID, std::shared_ptr<RegularizationGraph::Edge>>> good_edges;
    for (auto entry : edges) {
        if (entry.second->weight < min_weight_) {
            break;
        }

        good_edges.push_back(entry);
    }

    return good_edges;
}

bool RegularizationGraph::UpdateConnection(ID mappoint_id_1, ID mappoint_id_2,
                                           Eigen::Vector3f& landmark_position_1,
                                           Eigen::Vector3f& landmark_position_2) {
    if (!graph_.contains(mappoint_id_1) || !graph_.contains(mappoint_id_2)) {
        LOG(FATAL) << "Trying to update landmark that is not in the graph.";
        return false;
    }

    const float distance = (landmark_position_1 - landmark_position_2).norm();

    Eigen::Vector3f relative_position = landmark_position_2 - landmark_position_1;

    if (graph_[mappoint_id_1].contains(mappoint_id_2) &&
        graph_[mappoint_id_2].contains(mappoint_id_1)) {

        shared_ptr<RegularizationGraph::Edge> edge = graph_[mappoint_id_1][mappoint_id_2];

        if(distance > edge->max_distance) {
            edge->max_distance = distance;
        }
        if(distance < edge->min_distance) {
            edge->min_distance = distance;
        }

        edge->weight = InterpolationWeight(edge->max_distance, options_.weight_sigma);
        edge->last_relative_position = relative_position;

        // Edge prune criteria.
        if(fabs((edge->max_distance - edge->min_distance) / edge->min_distance) > options_.streching_th) {
            graph_[mappoint_id_1][mappoint_id_2]->status = BAD;

            return false;
        } else {
            return true;
        }
    } else {
        LOG(FATAL) << "Connection was not present in the graph.";
        return false;
    }
}

int RegularizationGraph::UpdateVertex(ID mappoint_id) {
    Eigen::Vector3f landmark_position = map_->GetMapPoint(mappoint_id)->GetLastWorldPosition();
    VertexConnections& connections = graph_[mappoint_id];

    int n_good_connections = 0;

    for (auto & [other_mappoint_id, edge] : connections){
        // Compute new distance.
        Eigen::Vector3f other_landmark_position = map_->GetMapPoint(other_mappoint_id)->GetLastWorldPosition();

        if (UpdateConnection(mappoint_id, other_mappoint_id,
                             landmark_position, other_landmark_position))
            n_good_connections++;
    }

    return n_good_connections;
}

std::string RegularizationGraph::ToString() {
    string graph_string;
    for (const auto &[id, connections] : graph_) {
        for (const auto &[id_other, edge] : connections) {
            graph_string += to_string(id) + " " + to_string(id_other) + " " + to_string(edge->weight) + "\n";
        }
    }

    return graph_string;
}

void RegularizationGraph::GetOptimizationNeighbours(const std::vector<ID>& mappoints_ids, const int connections_per_point,
                                                    absl::flat_hash_map<ID, absl::flat_hash_set<ID>>& zero_order_connections,
                                                    absl::flat_hash_map<ID, absl::flat_hash_set<ID>>& first_order_connections,
                                                    absl::flat_hash_set<ID>& second_order_connections) {
    for (auto mappoint_id : mappoints_ids) {
        zero_order_connections[mappoint_id] = absl::flat_hash_set<ID>();
    }

    for (auto mappoint_id : mappoints_ids) {
        auto connections =
                this->GetEdges(mappoint_id);

        int n_connections = 0;
        for(const auto& [mappoint_id_other, regularization_edge] : connections) {
            if (n_connections > connections_per_point ||
                regularization_edge->status == RegularizationGraph::BAD) {
                break;
            }

            if (zero_order_connections[mappoint_id].contains(mappoint_id_other)) {
                continue;
            }

            if (!zero_order_connections.contains(mappoint_id_other)) {
                if (!first_order_connections.contains(mappoint_id_other)) {
                    first_order_connections[mappoint_id_other] = absl::flat_hash_set<ID>();
                }

                continue;
            }

            zero_order_connections[mappoint_id].insert(mappoint_id_other);
            zero_order_connections[mappoint_id_other].insert(mappoint_id);

            n_connections++;
        }
    }

    for (auto &[mappoint_id, best_connections] : first_order_connections) {
        auto connections =
                this->GetEdges(mappoint_id);

        int n_connections = 0;
        for (const auto& [mappoint_id_other, regularization_edge] : connections) {
            if (n_connections > connections_per_point ||
                regularization_edge->status == RegularizationGraph::BAD) {
                break;
            }

            if (zero_order_connections.contains(mappoint_id_other)) {
                if (!zero_order_connections[mappoint_id_other].contains(mappoint_id)) {
                    zero_order_connections[mappoint_id_other].insert(mappoint_id);
                }

                continue;
            }

            if (first_order_connections[mappoint_id].contains(mappoint_id_other)) {
                continue;
            }

            if (!first_order_connections.contains(mappoint_id_other)) {
                first_order_connections[mappoint_id].insert(mappoint_id_other);
                second_order_connections.insert(mappoint_id_other);
                continue;
            }

            first_order_connections[mappoint_id].insert(mappoint_id_other);
            first_order_connections[mappoint_id_other].insert(mappoint_id);

            n_connections++;
        }
    }
}

float RegularizationGraph::GetMinWeightAllowed() {
    return min_weight_;
}
