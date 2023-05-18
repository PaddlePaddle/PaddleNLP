# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging

logger = logging.getLogger(__name__)

# class Telemetry:
#     """
#     Haystack reports anonymous usage statistics to support continuous software improvements for all its users.

#     You can opt-out of sharing usage statistics by manually setting the environment
#     variable `HAYSTACK_TELEMETRY_ENABLED` as described for different operating systems on the
#     [documentation page](https://docs.haystack.deepset.ai/docs/telemetry#how-can-i-opt-out).

#     Check out the documentation for more details: [Telemetry](https://docs.haystack.deepset.ai/docs/telemetry).
#     """

#     def __init__(self):
#         """
#         Initializes the telemetry. Loads the user_id from the config file,
#         or creates a new id and saves it if the file is not found.

#         It also collects system information which cannot change across the lifecycle
#         of the process (for example `is_containerized()`).
#         """
#         posthog.api_key = "phc_C44vUK9R1J6HYVdfJarTEPqVAoRPJzMXzFcj8PIrJgP"
#         posthog.host = "https://eu.posthog.com"

#         # disable posthog logging
#         for module_name in ["posthog", "backoff"]:
#             logging.getLogger(module_name).setLevel(logging.CRITICAL)
#             # Prevent module from sending errors to stderr when an exception is encountered during an emit() call
#             logging.getLogger(module_name).addHandler(logging.NullHandler())
#             logging.getLogger(module_name).propagate = False

#         self.user_id = None

#         if CONFIG_PATH.exists():
#             # Load the config file
#             try:
#                 with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
#                     config = yaml.safe_load(config_file)
#                     if "user_id" in config:
#                         self.user_id = config["user_id"]
#             except Exception as e:
#                 logger.debug("Telemetry could not read the config file %s", CONFIG_PATH, exc_info=e)
#         else:
#             # Create the config file
#             logger.info(
#                 "Haystack sends anonymous usage data to understand the actual usage and steer dev efforts "
#                 "towards features that are most meaningful to users. You can opt-out at anytime by manually "
#                 "setting the environment variable HAYSTACK_TELEMETRY_ENABLED as described for different "
#                 "operating systems in the [documentation page](https://docs.haystack.deepset.ai/docs/telemetry#how-can-i-opt-out). "
#                 "More information at [Telemetry](https://docs.haystack.deepset.ai/docs/telemetry)."
#             )
#             CONFIG_PATH.parents[0].mkdir(parents=True, exist_ok=True)
#             self.user_id = str(uuid.uuid4())
#             try:
#                 with open(CONFIG_PATH, "w") as outfile:
#                     yaml.dump({"user_id": self.user_id}, outfile, default_flow_style=False)
#             except Exception as e:
#                 logger.debug("Telemetry could not write config file to %s", CONFIG_PATH, exc_info=e)

#         self.event_properties = collect_static_system_specs()

#     def send_event(self, event_name: str, event_properties: Optional[Dict[str, Any]] = None):
#         """
#         Sends a telemetry event.

#         :param event_name: The name of the event to show in PostHog.
#         :param event_properties: Additional event metadata. These are merged with the
#             system metadata collected in __init__, so take care not to overwrite them.
#         """
#         event_properties = event_properties or {}
#         dynamic_specs = collect_dynamic_system_specs()
#         try:
#             posthog.capture(
#                 distinct_id=self.user_id,
#                 event=event_name,
#                 # loads/dumps to sort the keys
#                 properties=json.loads(
#                     json.dumps({**self.event_properties, **dynamic_specs, **event_properties}, sort_keys=True)
#                 ),
#             )
#         except Exception as e:
#             logger.debug("Telemetry couldn't make a POST request to PostHog.", exc_info=e)


# def send_event(event_name: str, event_properties: Optional[Dict[str, Any]] = None):
#     """
#     Send a telemetry event, if telemetry is enabled.
#     """
#     try:
#         if telemetry:
#             telemetry.send_event(event_name=event_name, event_properties=event_properties)
#     except Exception as e:
#         # Never let telemetry break things
#         logger.debug("There was an issue sending a '%s' telemetry event", event_name, exc_info=e)
