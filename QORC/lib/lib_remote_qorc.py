#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import torch
import perceval as pcvl

from perceval.runtime import RemoteConfig
from merlin.core.merlin_processor import MerlinProcessor


def _spin_until_with_ctrlc(
    pred, timeout_s: float = 10.0, sleep_s: float = 0.02
) -> bool:
    import time as _t

    start = _t.time()
    try:
        while not pred():
            if _t.time() - start > timeout_s:
                return False
            _t.sleep(sleep_s)
        return True
    except KeyboardInterrupt:
        return False


def remote_qorc_quantum_layer(
    train_tensor,
    val_tensor,
    test_tensor,
    qorc_quantum_layer,
    qpu_device_name,
    logger,
):
    qpu_device_nsample = 200
    # qpu_device_nsample = 5000

    # qpu_device_timeout = 12
    qpu_device_timeout = 60000  # roughly 15h

    # chunk_concurrency = 10
    chunk_concurrency = 20

    # max_batch_size    = 32
    # max_batch_size    = 64
    # max_batch_size    = 128
    # max_batch_size    = 1024
    # max_batch_size    = 10240  # 10k images à la fois => 7/8 batchs par run  => En pratique plus long
    max_batch_size = 102400  # 100k images à la fois => un seul batch

    logger.info("Call to remote_qorc_quantum_layer ")
    logger.info(
        "Using MerlinProcessor with remote_processor name: {}".format(qpu_device_name)
    )
    # logger.info("max_batch_size:{}".format(max_batch_size))

    qpu_device_name = qpu_device_name.lower()

    LOCAL_STR = ":local"
    if LOCAL_STR in qpu_device_name:
        qorc_quantum_layer.force_simulation = (
            True  # Force local computation of the Quantum Layer
        )
        qpu_device_name = qpu_device_name.replace(LOCAL_STR, "")
        logger.info(
            "'{}' détecté: Traitement local du remote processor".format(LOCAL_STR)
        )

    valid_qpu_device_name_list = [
        "sim:slos",
        "sim:ascella",
        "sim:belenos",
        "qpu:ascella",
        "qpu:belenos",
    ]

    if qpu_device_name not in valid_qpu_device_name_list:
        logger.info(
            "Error in qorc: remote_processor_type not recognized:{}".format(
                qpu_device_name
            )
        )
        raise
        return -1

    # Création du MerlinProcessor
    qorc_quantum_layer.eval()
    token = os.environ.get("QUANDELA_TOKEN", "").strip()
    RemoteConfig.set_token(token)
    remote_processor = pcvl.RemoteProcessor(qpu_device_name)
    proc = MerlinProcessor(
        remote_processor,
        chunk_concurrency=chunk_concurrency,
        max_batch_size=max_batch_size,
    )

    train_size = train_tensor.shape[0]
    val_size = val_tensor.shape[0]
    test_size = test_tensor.shape[0]
    data_tensor = torch.cat([train_tensor, val_tensor, test_tensor], dim=0)
    logger.info("data_tensor.shape:{}".format(str(data_tensor.shape)))

    match qpu_device_name:
        case "sim:slos":
            logger.info("qpu_device_name=sim:slos  - Calcule le train/val/test")
            time_cour = time.time()

            fut = proc.forward_async(
                qorc_quantum_layer, data_tensor, nsample=qpu_device_nsample
            )
            _spin_until_with_ctrlc(
                lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=qpu_device_timeout
            )
            processed_data_tensor = fut.wait()

            duration = time.time() - time_cour
            logger.info("Durée (s): {}".format(duration))

        case "sim:ascella" | "qpu:ascella":
            # Parralléliser les 3 jobs
            logger.info("qpu_device_name=sim:ascella  - Calcule le train/val/test")
            time_cour = time.time()

            # Adapter le circuit à la puce QPU "ascella"
            # remote_processor = pcvl.RemoteProcessor(qpu_device_name)
            specs = remote_processor.specs
            pcvl.pdisplay(specs["specific_circuit"])
            print(type(specs["specific_circuit"]))
            print(specs.keys())

            print("Platform constraints:")
            print(specs["constraints"])
            print("\nPlatform supported parameters:")
            print(specs["parameters"])
            # print("\nDescription, documentation:")
            # print(specs["description"])
            # print(specs["documentation"])

            def get_circuit_physical_depth(circuit: pcvl.Circuit):
                t = type(circuit)
                match t:
                    case pcvl.components.BS:
                        return 1, [1]
                    case pcvl.components.PS:
                        return 0, [0]
                    case pcvl.components.Unitary:
                        return 2 * circuit.m, [2, 2]
                    case pcvl.components.Circuit:
                        if circuit.is_composite():
                            depths = [0] * circuit.m
                            d_current = 0
                            for modes, comp in circuit._components:  # type: ignore[attr-defined]
                                # print(modes, comp)
                                d_current = max(depths[m] for m in modes)
                                add_depth = get_circuit_physical_depth(comp)
                                for m in modes:
                                    depths[m] = d_current + add_depth
                            d_current = max(depths[m] for m in modes)
                            return d_current, depths
                        else:
                            raise ValueError(
                                "Erreur dans get_circuit_physical_depth: Le circuit n'est pas composite."
                            )
                    case _:
                        raise ValueError(
                            f"Erreur dans get_circuit_physical_depth: Type de circuit non géré: {t}"
                        )
                raise ValueError("Erreur dans get_circuit_physical_depth (interne).")

            # TODO: Continuer ici, le code qui splitte le circuit en 2 parts
            def split_circuit_physical_depth(
                circuit: pcvl.Circuit, c_pre: pcvl.Circuit
            ):
                t = type(circuit)
                match t:
                    case pcvl.components.BS:
                        return 1, [1]
                    case pcvl.components.PS:
                        return 0, [0]
                    case pcvl.components.Unitary:
                        return 2 * circuit.m, [2, 2]
                    case pcvl.components.Circuit:
                        if circuit.is_composite():
                            depths = [0] * circuit.m
                            d_current = 0
                            for modes, comp in circuit._components:  # type: ignore[attr-defined]
                                # print(modes, comp)
                                d_current = max(depths[m] for m in modes)
                                add_depth = get_circuit_physical_depth(comp)
                                for m in modes:
                                    depths[m] = d_current + add_depth
                            d_current = max(depths[m] for m in modes)
                            return d_current, depths
                        else:
                            raise ValueError(
                                "Erreur dans get_circuit_physical_depth: Le circuit n'est pas composite."
                            )
                    case _:
                        raise ValueError(
                            f"Erreur dans get_circuit_physical_depth: Type de circuit non géré: {t}"
                        )
                raise ValueError("Erreur dans get_circuit_physical_depth (interne).")

            spec_circuit = specs["specific_circuit"]
            d_current, depths = get_circuit_physical_depth(spec_circuit)
            print("circuit depths:", d_current, depths)

            # Réussir à découper le circuit en 2, sur une colonne de phase shifters:
            # 1) Parcourir le circuit (parcours de graphe) et calculer la profondeur de chaque mode
            # 2) Avec le meme algorithme, découper le circuit en 2 sur la profondeur
            #    et s'assurer que les 2 circuits combinés redonnent la meme unitaire que le circuit seul (avec un dict sur les phases des PS)
            # 3) Adapter l'algorithme précédent pour avoir 3 circuits distincts (en séparant la colonne de PS)
            # 4) Construire la QuantumLayer à partir de ces derniers circuits

            exit()

            fut = proc.forward_async(
                qorc_quantum_layer, data_tensor, nsample=qpu_device_nsample
            )
            _spin_until_with_ctrlc(
                lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=qpu_device_timeout
            )
            processed_data_tensor = fut.wait()

            duration = time.time() - time_cour
            logger.info("Durée (s): {}".format(duration))

        case _:
            # Cas général: On lance les calculs par défaut
            logger.info(
                "Qorc: Traitement général (case else) du remote processor: {} - Calcule le train/val/test".format(
                    qpu_device_name
                )
            )
            time_cour = time.time()

            fut = proc.forward_async(
                qorc_quantum_layer, data_tensor, nsample=qpu_device_nsample
            )
            _spin_until_with_ctrlc(
                lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=qpu_device_timeout
            )
            processed_data_tensor = fut.wait()

            duration = time.time() - time_cour
            logger.info("Durée (s): {}".format(duration))

    train_data_qorc = processed_data_tensor[:train_size]
    val_data_qorc = processed_data_tensor[train_size : (train_size + val_size)]
    test_data_qorc = processed_data_tensor[-test_size:]

    return train_data_qorc, val_data_qorc, test_data_qorc
