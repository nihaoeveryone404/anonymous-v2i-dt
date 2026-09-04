"""V2X simulator extracted without changing the environment equations."""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VehicleToBSEnv:
    """
    车对基站通信环境
    8辆车，16个基站，每辆车选择3个基站
    每个智能体只观察自己3个基站的3个特征：距离、负载、SINR
    动作：功率比例（3维，和为1）+ 每条链路的数据包数量（离散0-10）
    """

    def __init__(self, seed=42, use_bad_initial_allocation=False):
        set_seed(seed)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.n_vehicles = 8
        self.n_basestations = 16
        self.n_selected_bs = 3

        self.vehicle_bs_mapping = {}
        all_bs_indices = list(range(self.n_basestations))
        for v in range(self.n_vehicles):
            self.vehicle_bs_mapping[v] = self.rng.choice(
                all_bs_indices, self.n_selected_bs, replace=False
            ).tolist()

        print("车辆-基站分配映射:")
        for v, bs_list in self.vehicle_bs_mapping.items():
            print(f"  车辆{v}: 基站{bs_list}")

        self.episode_length = 20
        self.bandwidth = 5e6
        self.power_budget_per_vehicle_mw = 200.0
        self.total_packet_budget = 10
        self.max_packet_count = self.total_packet_budget
        self.n_packet_choices = self.max_packet_count + 1
        self.use_bad_initial_allocation = use_bad_initial_allocation

        self.bs_positions = self.rng.uniform(0, 1000, (self.n_basestations, 2))
        self.vehicle_positions = self.rng.uniform(0, 1000, (self.n_vehicles, 2))

        self.base_station_states = {}
        for i in range(self.n_basestations):
            self.base_station_states[f"BS{i}"] = {
                "position": self.bs_positions[i],
                "traffic_load": self.rng.uniform(0.1, 0.9),
                "sinr": 0.1,
            }

        self.continuous_action_dim = self.n_selected_bs
        self.discrete_action_dim = self.n_selected_bs
        self.obs_dim = self.n_selected_bs * 3
        self.state_dim = self.n_basestations * 4 + self.n_vehicles * 2
        self.current_step = 0
        self.first_allocation_done = not self.use_bad_initial_allocation
        self.link_shadow_fading_db = np.zeros((self.n_vehicles, self.n_basestations), dtype=np.float32)

    def reset(self):
        self.current_step = 0

        self.vehicle_positions += self.rng.uniform(-30, 30, (self.n_vehicles, 2))
        self.vehicle_positions = np.clip(self.vehicle_positions, 0, 1000)

        for i in range(self.n_basestations):
            traffic_change = self.rng.uniform(-0.1, 0.1)
            new_traffic = self.base_station_states[f"BS{i}"]["traffic_load"] + traffic_change
            self.base_station_states[f"BS{i}"]["traffic_load"] = np.clip(new_traffic, 0.1, 0.9)

        self.link_shadow_fading_db = self.rng.normal(0, 6, (self.n_vehicles, self.n_basestations)).astype(np.float32)
        self._refresh_bs_sinr_estimates()

        return self.get_obs(), self.get_state()

    def calculate_distance(self, vehicle_pos, bs_pos):
        return np.linalg.norm(vehicle_pos - bs_pos)

    def calculate_channel_gain(self, vehicle_idx, bs_idx, distance):
        distance = max(distance, 1.0)
        path_loss_db = 100 + 35 * np.log10(distance)
        shadow_fading = float(self.link_shadow_fading_db[vehicle_idx, bs_idx])
        total_loss_db = path_loss_db + shadow_fading
        return 10 ** (-total_loss_db / 10)

    def calculate_sinr(self, channel_gain, transmit_power_mw):
        noise_power_dbm = -174 + 10 * np.log10(self.bandwidth)
        noise_power_mw = 10 ** (noise_power_dbm / 10)
        signal_power_mw = transmit_power_mw * channel_gain
        sinr = signal_power_mw / (noise_power_mw + 1e-10)
        return float(np.clip(sinr, 0.1, 100.0))

    def _refresh_bs_sinr_estimates(self):
        for bs_idx in range(self.n_basestations):
            estimates = []
            for vehicle_idx in range(self.n_vehicles):
                if bs_idx not in self.vehicle_bs_mapping[vehicle_idx]:
                    continue
                distance = self.calculate_distance(
                    self.vehicle_positions[vehicle_idx],
                    self.base_station_states[f"BS{bs_idx}"]["position"],
                )
                gain = self.calculate_channel_gain(vehicle_idx, bs_idx, distance)
                nominal_power = self.power_budget_per_vehicle_mw / self.n_selected_bs
                estimates.append(self.calculate_sinr(gain, nominal_power))
            self.base_station_states[f"BS{bs_idx}"]["sinr"] = float(np.mean(estimates)) if estimates else 0.1

    def project_packet_counts(self, raw_packet_actions):
        """
        将3个离散动作(每个0~10)投影为总和固定为10的整数包分配。
        这样保留 v3 的多离散结构，同时满足单车总包数固定。
        """
        raw = np.asarray(raw_packet_actions, dtype=np.float32).reshape(-1)
        if raw.size != self.n_selected_bs:
            raw = np.resize(raw, self.n_selected_bs)
        raw = np.clip(raw, 0, self.max_packet_count)

        if float(np.sum(raw)) <= 1e-6:
            scaled = np.ones(self.n_selected_bs, dtype=np.float32) * (self.total_packet_budget / self.n_selected_bs)
        else:
            scaled = raw / float(np.sum(raw)) * self.total_packet_budget

        packet_counts = np.floor(scaled).astype(int)
        remainder = int(self.total_packet_budget - np.sum(packet_counts))
        if remainder > 0:
            fractional = scaled - packet_counts
            order = np.argsort(-fractional)
            for idx in order[:remainder]:
                packet_counts[idx] += 1
        elif remainder < 0:
            order = np.argsort(-(packet_counts - scaled))
            for idx in order[:(-remainder)]:
                if packet_counts[idx] > 0:
                    packet_counts[idx] -= 1

        diff = int(self.total_packet_budget - np.sum(packet_counts))
        if diff != 0:
            packet_counts[0] += diff

        packet_counts = np.clip(packet_counts, 0, self.total_packet_budget)
        return packet_counts.astype(int)

    def calculate_delay(self, vehicle_idx, bs_idx, packet_count, power_ratio):
        if packet_count <= 0:
            return 0.0

        vehicle_pos = self.vehicle_positions[vehicle_idx]
        bs_pos = self.base_station_states[f"BS{bs_idx}"]["position"]
        distance = self.calculate_distance(vehicle_pos, bs_pos)

        channel_gain = self.calculate_channel_gain(vehicle_idx, bs_idx, distance)
        transmit_power_mw = max(power_ratio, 1e-6) * self.power_budget_per_vehicle_mw
        sinr = self.calculate_sinr(channel_gain, transmit_power_mw)
        self.base_station_states[f"BS{bs_idx}"]["sinr"] = sinr

        data_rate = self.bandwidth * np.log2(1 + sinr)
        packet_size_bits = 1500 * 8
        total_bits = packet_count * packet_size_bits

        transmission_delay_ms = (total_bits / max(data_rate, 1e-6)) * 1000.0
        processing_delay_ms = packet_count * 0.5
        queueing_delay_ms = self.base_station_states[f"BS{bs_idx}"]["traffic_load"] * 3.0

        total_delay_ms = transmission_delay_ms + processing_delay_ms + queueing_delay_ms
        return float(total_delay_ms)

    def calculate_reward(self, system_delay, avg_vehicle_max_delay, total_power_ratio, avg_packets):
        del system_delay, total_power_ratio, avg_packets
        return float(-avg_vehicle_max_delay)

    def step(self, continuous_actions, discrete_actions):
        continuous_actions = np.array(continuous_actions, dtype=np.float32)
        discrete_actions = np.array(discrete_actions, dtype=np.int32)

        if continuous_actions.ndim == 1:
            continuous_actions = continuous_actions.reshape(1, -1)
        if discrete_actions.ndim == 1:
            discrete_actions = discrete_actions.reshape(1, -1)

        vehicle_delays = []
        vehicle_power_usage = []
        vehicle_packet_usage = []
        vehicle_peak_power_usage = []
        vehicle_peak_packet_usage = []

        for vehicle_idx in range(self.n_vehicles):
            bs_indices = self.vehicle_bs_mapping[vehicle_idx]

            if vehicle_idx < len(continuous_actions):
                power_ratios = np.clip(continuous_actions[vehicle_idx].flatten(), 1e-6, 1.0)
                power_ratios = power_ratios / max(np.sum(power_ratios), 1e-6)
            else:
                power_ratios = np.ones(self.n_selected_bs, dtype=np.float32) / self.n_selected_bs

            if vehicle_idx < len(discrete_actions):
                packet_counts = self.project_packet_counts(discrete_actions[vehicle_idx].flatten())
            else:
                packet_counts = self.project_packet_counts(np.ones(self.n_selected_bs, dtype=int))

            if self.current_step == 0 and not self.first_allocation_done:
                worst_idx = int(np.argmax([
                    self.calculate_distance(
                        self.vehicle_positions[vehicle_idx],
                        self.base_station_states[f"BS{bs_idx}"]["position"],
                    ) + 200 * self.base_station_states[f"BS{bs_idx}"]["traffic_load"]
                    for bs_idx in bs_indices
                ]))
                power_ratios = np.ones(self.n_selected_bs, dtype=np.float32) * 0.15
                power_ratios[worst_idx] = 0.70
                power_ratios = power_ratios / power_ratios.sum()
                packet_counts = np.zeros(self.n_selected_bs, dtype=int)
                packet_counts[worst_idx] = self.total_packet_budget

            vehicle_power_usage.append(float(np.sum(power_ratios)))
            vehicle_packet_usage.append(int(np.sum(packet_counts)))
            vehicle_peak_power_usage.append(float(np.max(power_ratios)))
            vehicle_peak_packet_usage.append(int(np.max(packet_counts)))

            bs_delays = []
            for i, bs_idx in enumerate(bs_indices):
                delay = self.calculate_delay(
                    vehicle_idx=vehicle_idx,
                    bs_idx=bs_idx,
                    packet_count=int(packet_counts[i]),
                    power_ratio=float(power_ratios[i]),
                )
                if packet_counts[i] > 0:
                    bs_delays.append(delay)

            vehicle_delays.append(max(bs_delays) if bs_delays else 0.1)

        system_delay = float(max(vehicle_delays))
        avg_vehicle_max_delay = float(np.mean(vehicle_delays))
        avg_power_ratio = float(np.mean(vehicle_power_usage))
        avg_packets = float(np.mean(vehicle_packet_usage))
        avg_peak_power_ratio = float(np.mean(vehicle_peak_power_usage))
        avg_peak_packets = float(np.mean(vehicle_peak_packet_usage))
        reward = self.calculate_reward(
            system_delay=system_delay,
            avg_vehicle_max_delay=avg_vehicle_max_delay,
            total_power_ratio=avg_power_ratio,
            avg_packets=avg_packets,
        )

        self.current_step += 1
        done = self.current_step >= self.episode_length
        if done and not self.first_allocation_done:
            self.first_allocation_done = True
            print("✅ 第一个episode完成，已标记初始分配完成")

        self.vehicle_positions += self.rng.uniform(-20, 20, (self.n_vehicles, 2))
        self.vehicle_positions = np.clip(self.vehicle_positions, 0, 1000)
        self._refresh_bs_sinr_estimates()

        info = {
            "system_delay_ms": system_delay,
            "avg_vehicle_max_delay_ms": avg_vehicle_max_delay,
            "power_usage": vehicle_power_usage,
            "packet_usage": vehicle_packet_usage,
            "peak_power_usage": vehicle_peak_power_usage,
            "peak_packet_usage": vehicle_peak_packet_usage,
            "avg_peak_power_usage": avg_peak_power_ratio,
            "avg_peak_packet_usage": avg_peak_packets,
        }
        return self.get_obs(), self.get_state(), reward, done, info

    def get_obs(self):
        obs = []
        for vehicle_idx in range(self.n_vehicles):
            vehicle_obs = []
            for bs_idx in self.vehicle_bs_mapping[vehicle_idx]:
                bs_state = self.base_station_states[f"BS{bs_idx}"]
                distance = self.calculate_distance(
                    self.vehicle_positions[vehicle_idx], bs_state["position"]
                )
                distance_norm = distance / 1000.0
                traffic_load = float(bs_state["traffic_load"])
                sinr_norm = min(float(bs_state["sinr"]) / 100.0, 1.0)
                vehicle_obs.extend([distance_norm, traffic_load, sinr_norm])
            obs.append(np.array(vehicle_obs, dtype=np.float32))
        return obs

    def get_state(self):
        state = []
        for bs_idx in range(self.n_basestations):
            bs_state = self.base_station_states[f"BS{bs_idx}"]
            pos_norm = bs_state["position"] / 1000.0
            traffic_load = float(bs_state["traffic_load"])
            sinr_norm = min(float(bs_state["sinr"]) / 100.0, 1.0)
            state.extend([pos_norm[0], pos_norm[1], traffic_load, sinr_norm])

        for vehicle_idx in range(self.n_vehicles):
            vpos_norm = self.vehicle_positions[vehicle_idx] / 1000.0
            state.extend([float(vpos_norm[0]), float(vpos_norm[1])])

        return np.array(state, dtype=np.float32)
