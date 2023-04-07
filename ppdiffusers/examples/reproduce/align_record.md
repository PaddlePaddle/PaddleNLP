

## 示例prompt

以[Hugging Face Spaces](https://huggingface.co/spaces/AttendAndExcite/Attend-and-Excite)选用的prompt和seed为例展示对齐效果如下表：`image_error`一栏分别展示了ppdiffusers和diffusers生成的图片及其绝对误差和相对误差在像素点的分布；`latents_error`一栏展示了模型在去噪过程中每一步生成的潜变量的最大绝对误差（蓝色线条）和最大相对误差（橙色线条）。

| prompt                                                       | seed | image_error                                                  | latents_error                                                |
| :----------------------------------------------------------- | :--: | ------------------------------------------------------------ | ------------------------------------------------------------ |
| A grizzly bear catching a salmon in a crystal clear river surrounded by a forest | 123  | ![123_error_map](https://user-images.githubusercontent.com/40912707/226089730-215319e0-e9f5-4b7a-b51d-593b82a2e0a2.png) | ![123](https://user-images.githubusercontent.com/40912707/226089728-52b2f185-8cf1-4f39-96a2-9d9bae1e2baf.png) |
| A horse and a dog                                            | 123  | ![123_error_map](https://user-images.githubusercontent.com/40912707/226089753-fa78e6d8-c241-4436-be1b-e63ac8318199.png)             | !![123](https://user-images.githubusercontent.com/40912707/226089754-18c1b9ce-0287-4f04-8b85-4dbcdb0871c8.png)                       |
| A mouse and a red car                                        | 2098 | ![2098_error_map](https://user-images.githubusercontent.com/40912707/226089872-9a6fa7a7-f8e5-45fd-acd2-a9ae5e7df6af.png)         | ![2098](https://user-images.githubusercontent.com/40912707/226089874-7004bb01-5e19-4c5a-bd55-af741d9e1e38.png)                  |
| A painting of an elephant with glasses                       | 123  | ![123_error_map](https://user-images.githubusercontent.com/40912707/226089922-3d833b5c-a522-47b2-a9ed-5d95f2e4b411.png) | ![123](https://user-images.githubusercontent.com/40912707/226089923-35513880-6de1-4da9-bd57-4c747471c359.png)   |
| A playful kitten chasing a butterfly in a wildflower meadow  | 123  | ![123_error_map](https://user-images.githubusercontent.com/40912707/226089933-5fd1408a-ea13-4329-8f86-391fad46cce8.png) | ![123](https://user-images.githubusercontent.com/40912707/226089934-ce392b7a-e672-46d2-a5e3-d66d23dc66ed.png) |
| A pod of dolphins leaping out of the water in an ocean with a ship on the background | 123  | ![123_error_map](https://user-images.githubusercontent.com/40912707/226089943-b5def711-deb0-4cb2-9589-79693fd46a4f.png) | ![123](https://user-images.githubusercontent.com/40912707/226089945-3e7c37a4-17d5-4d26-99ee-40812957dea3.png) |





## 更多的prompt


按照论文的prompt构造方法尝试更多的prompt和seed对比ppdiffuser和diffusers的表现。

三种prompt模板分别如下，加粗的token表示使用注意力的token：

*   a **{animal}** and a **{animal}**
*   a **{animal}** and a {color} **{object}**
*   a {colorA} **{objectA}** and a {colorB} **{objectB}**

其中用于填充的单词如下表

| category | words                                                        |
| -------- | ------------------------------------------------------------ |
| animals  | cat, dog, bird, bear, lion, horse, elephant, monkey, frog, turtle, rabbit, mouse |
| objects  | backpack, glasses, crown, suitcase, chair, balloon, bow, car, bowl, bench, clock, apple |
| colors   | red, orange, yellow, green, blue, purple, pink, brown, gray, black, white |

对每一个prompt再随机生成5个seed，对比结果绘制散点图如下，其中每个点代表该prompt和seed的设置下在该step上的误差，颜色越亮则表示散点越聚集：

| 绝对误差                   | 相对误差                   |
| -------------------------- | -------------------------- |
| ![align_record_atol](https://user-images.githubusercontent.com/40912707/226089978-a760e900-9309-4058-9075-b5853fcda549.png) | ![align_record_rtol](https://user-images.githubusercontent.com/40912707/226089977-93d9e502-73ce-4b53-ab3d-066f603ce8d6.png) |

在测试的105组prompt和seed中，有13个prompt和seed的组合生成的图片有较大差别，记录如下：

| prompt                                 | seed | image_error                                                  | latents_error                                               |
| :------------------------------------- | ---- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| a dog and a pink crown                 | 1930 | ![1930_error_map](https://user-images.githubusercontent.com/40912707/226090081-1448b7af-db20-4650-b181-44d02421a8fd.png)        | ![1930](https://user-images.githubusercontent.com/40912707/226090079-52dfc07d-db0b-46b6-a870-dee4304a19de.png)                 |
| a mouse and a yellow apple             | 28   | ![28_error_map](https://user-images.githubusercontent.com/40912707/226090224-ccda6bfc-d0c2-4332-baac-be4ca92f09ae.png)      | ![28](https://user-images.githubusercontent.com/40912707/226090236-3d362322-c6c3-4d9c-9769-60bdf554c030.png)               |
| a dog and a pink crown                 | 1285 | ![1285_error_map](https://user-images.githubusercontent.com/40912707/226090083-1a654c9f-8c79-45c6-ac4c-88065a9526a0.png)        | ![1285](https://user-images.githubusercontent.com/40912707/226090082-134c5aa3-7c8d-4481-acde-f4303db3f8d5.png)                 |
| a mouse and a yellow apple             | 60   | ![60_error_map](https://user-images.githubusercontent.com/40912707/226090232-9b5e09d8-139b-4b75-9bd8-463443cb57e2.png)      |![60](https://user-images.githubusercontent.com/40912707/226090231-dcd6aadb-185d-49c8-925f-ef4c491a4197.png)            |
| a mouse and a yellow apple             | 49   | ![49_error_map](https://user-images.githubusercontent.com/40912707/226090227-0d760233-c9d4-4a54-88c5-cb044ae15571.png)      | ![49](https://user-images.githubusercontent.com/40912707/226090226-f31ca86d-5947-471c-904c-173d5fd515ad.png)   |
| a white chair and a green chair        | 933  | ![933_error_map](https://user-images.githubusercontent.com/40912707/226090329-c585ffcf-e17a-4cd1-94e1-ac91f3354f0b.png) | ![933](https://user-images.githubusercontent.com/40912707/226090328-6fab40a9-d9e8-457d-8d99-d37bf860f215.png) |
| a dog and a black bow                  | 148  | ![148_error_map](https://user-images.githubusercontent.com/40912707/226090404-344e8004-ee42-454a-9b08-de19bfc85ad8.png) | ![148](https://user-images.githubusercontent.com/40912707/226090406-d03b9529-7667-4520-b6be-fbbd558230ae.png) |
| A horse and a dog                      | 27   | ![27_error_map](https://user-images.githubusercontent.com/40912707/226090430-0827b75c-19b6-4357-b475-28edf4b8cadc.png) | ![27](https://user-images.githubusercontent.com/40912707/226090432-4523dc64-9ce7-42ef-bdf2-8e192cebb212.png)|
| a horse and a gray bow                 | 1753 | !![1743_error_map](https://user-images.githubusercontent.com/40912707/226090077-637bcfff-cbfa-44c9-a5c9-6a482526a734.png)        | ![1743](https://user-images.githubusercontent.com/40912707/226090074-e84b8ab2-27db-444a-9f73-142e91e3a286.png)                 |
| A painting of an elephant with glasses | 1860 | ![1860_error_map](https://user-images.githubusercontent.com/40912707/226090464-2554b7ca-c40b-4b8b-b746-1529050e91c7.png) |![1860](https://user-images.githubusercontent.com/40912707/226090462-c67f7210-9026-49bc-bf6d-b70dc626be46.png)|
| a horse and a gray bow                 | 1597 | ![1597_error_map](https://user-images.githubusercontent.com/40912707/226090499-c7ae25d1-59b7-4bfb-b318-048e6c180a92.png) |![1597](https://user-images.githubusercontent.com/40912707/226090498-f2fc3617-843f-4968-8369-3db92b2ad8b5.png)  |
| a dog and a pink crown                 | 1743 | ![1743_error_map](https://user-images.githubusercontent.com/40912707/226090529-b56f5e0c-0808-4dfc-b7be-d7f0dfa01696.png)| ![1743](https://user-images.githubusercontent.com/40912707/226090528-5425e457-11a6-4985-b17b-237eae93c138.png) |
| a white chair and a green chair        | 890  |![890_error_map](https://user-images.githubusercontent.com/40912707/226090327-929f60ca-5eb3-4019-a19b-db1ad8a307d2.png) | ![890](https://user-images.githubusercontent.com/40912707/226090332-d8b281ed-2a2b-4964-8c07-e0120990b344.png)   |



#### 由于存在梯度，同样的prompt和seed设置下多次运行依旧存在误差

| diffusers                                                    | ppdiffusers                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20230318002849117](https://user-images.githubusercontent.com/40912707/226090011-7d4fb85b-229f-4d62-b6bf-b7166c9341ef.png) | ![image-20230318002900522](https://user-images.githubusercontent.com/40912707/226090012-1ffe4095-fa18-46fb-8b11-f9b8369c79a6.png) |
