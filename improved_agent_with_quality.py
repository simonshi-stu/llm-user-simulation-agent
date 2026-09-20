"""
增强版Agent - 在推理过程中考虑useful/funny/cool因素
虽然最终只返回stars和review，但在生成过程中会考虑评论的这些特性
"""

import re
from collections import Counter
import logging

# 尝试从 websocietysimulator 导入（如果环境完整，这是优先方案）
try:
    from websocietysimulator.agent import SimulationAgent
    from websocietysimulator.llm import LLMBase
    from websocietysimulator.agent.modules.planning_modules import PlanningBase
    from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
    from websocietysimulator.agent.modules.memory_modules import MemoryDILU
    logging.info("Using websocietysimulator base classes.")
except Exception as e:
    # 如果导入失败（比如本地没有安装 websocietysimulator），
    # 就用本地的简化版基类，这样这个文件仍然可以被 import。
    logging.warning(f"Failed to import websocietysimulator, using local stubs instead: {e}")

    class SimulationAgent:
        def __init__(self, llm):
            self.llm = llm
            self.interaction_tool = None
            self.task = {}

    class LLMBase:
        def __call__(self, messages, **kwargs):
            raise NotImplementedError("LLMBase.__call__ is not implemented.")

    class PlanningBase:
        def __init__(self, llm=None):
            self.llm = llm

    class ReasoningBase:
        def __init__(self, profile_type_prompt: str = "", memory=None, llm=None):
            self.llm = llm
            self.memory = memory
            self.profile_type_prompt = profile_type_prompt

    class MemoryDILU:
        def __init__(self, llm=None):
            self.llm = llm

        def __call__(self, memory_str: str):
            # 简单 stub，什么都不做
            pass

class EnhancedPlanning(PlanningBase):
    """改进的规划模块"""

    def __init__(self, llm):
        super().__init__(llm=llm)

    def __call__(self, task_description):
        self.plan = [
            {
                'description': 'Retrieve user profile and historical reviews',
                'reasoning instruction': 'Analyze user characteristics and review patterns',
                'tool use instruction': task_description['user_id']
            },
            {
                'description': 'Retrieve business information and category',
                'reasoning instruction': 'Understand business type and attributes',
                'tool use instruction': task_description['item_id']
            },
            {
                'description': 'Collect and analyze existing reviews for this business',
                'reasoning instruction': 'Identify common themes and user sentiments',
                'tool use instruction': f"reviews_for_{task_description['item_id']}"
            }
        ]
        return self.plan


class UserProfileAnalyzer:
    """用户画像分析器 - 包含useful/funny/cool特征分析"""

    @staticmethod
    def analyze_user_patterns(reviews_user):
        """深度分析用户的评论模式，包括评论特征"""
        if not reviews_user or len(reviews_user) == 0:
            return {
                'user_type': '新用户',
                'avg_stars': 3.0,
                'avg_length': 100,
                'rating_tendency': 'neutral',
                'star_distribution': {},
                'review_count': 0,
                'useful_tendency': 'low',      # 新增
                'funny_tendency': 'low',        # 新增
                'engagement_style': 'neutral'   # 新增
            }

        # 基本统计
        stars_list = [r['stars'] for r in reviews_user]
        avg_stars = sum(stars_list) / len(stars_list)

        lengths = [len(r['text']) for r in reviews_user]
        avg_length = sum(lengths) / len(lengths)

        star_distribution = Counter(stars_list)

        # 分析useful/funny/cool特征（如果数据中有这些字段）
        useful_scores = []
        funny_scores = []
        cool_scores = []

        for r in reviews_user:
            # 尝试获取这些字段，如果没有则设为0
            useful_scores.append(r.get('useful', 0))
            funny_scores.append(r.get('funny', 0))
            cool_scores.append(r.get('cool', 0))

        # 计算平均值
        avg_useful = sum(useful_scores) / len(useful_scores) if useful_scores else 0
        avg_funny = sum(funny_scores) / len(funny_scores) if funny_scores else 0
        avg_cool = sum(cool_scores) / len(cool_scores) if cool_scores else 0

        # 判断用户的评论特征倾向
        if avg_useful > 2.0:
            useful_tendency = 'high'  # 写有用的信息性评论
        elif avg_useful > 0.5:
            useful_tendency = 'medium'
        else:
            useful_tendency = 'low'

        if avg_funny > 0.5:
            funny_tendency = 'high'  # 幽默风趣的评论
        elif avg_funny > 0.1:
            funny_tendency = 'medium'
        else:
            funny_tendency = 'low'

        # 综合判断engagement风格
        if avg_useful > 1.0 and avg_length > 150:
            engagement_style = 'informative'  # 信息丰富型
        elif avg_funny > 0.3:
            engagement_style = 'entertaining'  # 娱乐型
        elif avg_cool > 0.5:
            engagement_style = 'insightful'  # 有洞察力型
        else:
            engagement_style = 'straightforward'  # 直接简洁型

        # 评分倾向
        if avg_stars >= 4.0:
            rating_tendency = 'positive'
            user_type = '乐观型用户（倾向给高分）'
        elif avg_stars <= 2.5:
            rating_tendency = 'critical'
            user_type = '挑剔型用户（评分严格）'
        else:
            rating_tendency = 'neutral'
            user_type = '中立型用户（评分客观）'

        review_style = 'detailed' if avg_length > 150 else 'concise'

        return {
            'user_type': user_type,
            'avg_stars': round(avg_stars, 2),
            'avg_length': int(avg_length),
            'rating_tendency': rating_tendency,
            'star_distribution': dict(star_distribution),
            'review_count': len(reviews_user),
            'review_style': review_style,
            # 新增的特征
            'avg_useful': round(avg_useful, 2),
            'avg_funny': round(avg_funny, 2),
            'avg_cool': round(avg_cool, 2),
            'useful_tendency': useful_tendency,
            'funny_tendency': funny_tendency,
            'engagement_style': engagement_style
        }

    @staticmethod
    def format_user_analysis(user_profile):
        """格式化用户分析结果为prompt"""

        # 根据engagement_style给出具体的风格描述
        style_descriptions = {
            'informative': '信息丰富、详细具体，经常被认为有用',
            'entertaining': '幽默风趣、生动有趣',
            'insightful': '有深度、有见地的评论',
            'straightforward': '直接简洁、实用'
        }

        style_desc = style_descriptions.get(
            user_profile['engagement_style'],
            '一般风格'
        )

        return f"""
用户特征分析：
- 用户类型：{user_profile['user_type']}
- 历史评分均值：{user_profile['avg_stars']}星
- 评论数量：{user_profile['review_count']}条
- 评论风格：{'详细型' if user_profile['review_style'] == 'detailed' else '简洁型'}（平均{user_profile['avg_length']}字）
- 评分分布：{user_profile['star_distribution']}

评论特征（这些特征会影响你的评论风格）：
- 评论风格类型：{style_desc}
- 信息价值倾向：{user_profile['useful_tendency']} (历史评论平均获得{user_profile['avg_useful']}个useful标记)
- 幽默程度：{user_profile['funny_tendency']} (历史评论平均获得{user_profile['avg_funny']}个funny标记)
"""


class ReviewQualityAnalyzer:
    """分析其他评论的质量特征，用于指导生成"""

    @staticmethod
    def analyze_review_qualities(reviews):
        """
        分析一组评论的质量特征

        Returns:
            dict: 包含useful/funny/cool的统计信息
        """
        if not reviews:
            return {
                'has_useful_examples': False,
                'has_funny_examples': False,
                'common_themes': []
            }

        useful_reviews = []
        funny_reviews = []

        for review in reviews:
            useful_count = review.get('useful', 0)
            funny_count = review.get('funny', 0)

            if useful_count > 2:  # 被标记为有用的评论
                useful_reviews.append({
                    'text': review['text'][:200],
                    'stars': review.get('stars', 'N/A'),
                    'useful': useful_count
                })

            if funny_count > 1:  # 被标记为有趣的评论
                funny_reviews.append({
                    'text': review['text'][:200],
                    'stars': review.get('stars', 'N/A'),
                    'funny': funny_count
                })

        return {
            'has_useful_examples': len(useful_reviews) > 0,
            'has_funny_examples': len(funny_reviews) > 0,
            'useful_reviews': useful_reviews[:2],  # 最多2条
            'funny_reviews': funny_reviews[:2],    # 最多2条
            'total_reviews': len(reviews)
        }


class ReasoningWithQualityAwareness(ReasoningBase):
    """考虑评论质量特征的推理模块"""

    def __init__(self, profile_type_prompt, llm):
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str, enable_reflection: bool = True):
        """
        两阶段推理：生成初稿 + 反思改进
        在生成过程中考虑useful/funny/cool特征
        """

        # 第一阶段：生成初稿
        draft_prompt = f'''{task_description}

请生成评论初稿。注意：
1. 根据你的历史评论特征（信息性/娱乐性/洞察力），调整评论风格
2. 如果你的评论通常被认为有用，那就多写具体细节和实用信息
3. 如果你的评论通常比较幽默，可以适当加入轻松的语气
4. 保持与你历史风格的一致性

严格按照以下格式输出：
stars: [1.0/2.0/3.0/4.0/5.0]
review: [你的评论文本]
'''

        messages = [{"role": "user", "content": draft_prompt}]
        draft_result = self.llm(
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )

        if not enable_reflection:
            return draft_result

        # 第二阶段：质量反思
        reflection_prompt = f'''
你刚刚生成了这个评论：

{draft_result}

请从以下几个方面进行质量评估和改进：

1. **信息价值** (Usefulness)：
   - 评论是否提供了具体、实用的信息？
   - 是否帮助其他用户做决策？
   - 是否包含具体的细节（如菜品名称、价格、服务细节等）？

2. **可读性和趣味性**：
   - 评论是否自然流畅？
   - 如果用户历史风格偏幽默，是否体现了这一点？
   - 语气是否符合用户的历史风格？

3. **风格一致性**：
   - 评论长度是否与用户历史习惯一致？
   - 评分是否符合用户的评分倾向？
   - 详细程度是否匹配用户的typical风格？

4. **真实性和具体性**：
   - 评论是否像真实用户写的？
   - 是否避免了过于模板化的表达？
   - 是否提到了具体的体验细节？

基于以上分析，提供改进后的版本：

严格按照以下格式输出：
stars: [1.0/2.0/3.0/4.0/5.0]
review: [改进后的评论，确保信息价值高、风格一致]
'''

        messages = [{"role": "user", "content": reflection_prompt}]
        final_result = self.llm(
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )

        return final_result


class ImprovedSimulationAgent(SimulationAgent):
    """
    增强版Agent - 在推理过程中考虑useful/funny/cool等质量因素
    最终返回：{"stars": float, "review": str}
    """

    def __init__(self, llm: LLMBase, enable_reflection: bool = True,
                 use_memory: bool = True, max_reference_reviews: int = 5):
        super().__init__(llm=llm)

        self.enable_reflection = enable_reflection
        self.use_memory = use_memory
        self.max_reference_reviews = max_reference_reviews

        self.planning = EnhancedPlanning(llm=self.llm)
        self.reasoning = ReasoningWithQualityAwareness(
            profile_type_prompt='',
            llm=self.llm
        )

        if self.use_memory:
            self.memory = MemoryDILU(llm=self.llm)

        self.profile_analyzer = UserProfileAnalyzer()
        self.quality_analyzer = ReviewQualityAnalyzer()

    def parse_review_result(self, result: str):
        """解析LLM输出"""
        try:
            lines = result.strip().split('\n')
            stars = None
            review_text = None

            for line in lines:
                line_stripped = line.strip()
                line_lower = line_stripped.lower()

                if 'stars:' in line_lower or '星级:' in line_lower:
                    match = re.search(r'(\d+\.?\d*)', line_stripped)
                    if match:
                        stars = float(match.group(1))

                elif 'review:' in line_lower or '评论:' in line_lower:
                    parts = line_stripped.split(':', 1)
                    if len(parts) >= 2:
                        review_text = parts[1].strip()

            if stars is None:
                logging.warning(f"无法解析stars，使用默认值3.0")
                stars = 3.0

            if review_text is None:
                logging.warning(f"无法解析review，使用默认文本")
                review_text = "不错的体验。"

            stars = float(stars)
            if stars not in [1.0, 2.0, 3.0, 4.0, 5.0]:
                stars = round(stars)
                stars = max(1.0, min(5.0, float(stars)))

            return stars, review_text

        except Exception as e:
            logging.error(f'解析错误: {e}')
            return 3.0, "一般的体验。"

    def get_relevant_reviews(self, reviews_item, top_k: int = 5):
        """获取相关评论"""
        if not reviews_item:
            return []

        valuable_reviews = [
            r for r in reviews_item
            if len(r.get('text', '')) > 50
        ]

        if len(valuable_reviews) < 3:
            valuable_reviews = reviews_item

        return valuable_reviews[:top_k]

    def build_prompt(self, user_info, business_info, user_profile_analysis,
                     reference_reviews, user_recent_review, quality_analysis):
        """构建包含质量意识的prompt"""

        # 格式化参考评论，特别标注高质量评论
        reference_text = ""
        if reference_reviews:
            reference_text = "其他用户对这家商家的评论：\n"

            # 如果有被标记为useful的评论，特别指出
            if quality_analysis['has_useful_examples']:
                reference_text += "\n【信息价值高的评论示例】：\n"
                for i, review in enumerate(quality_analysis['useful_reviews'], 1):
                    reference_text += f"{i}. [{review['stars']}星, {review['useful']}人认为有用] {review['text']}\n"

            # 如果有被标记为funny的评论，特别指出
            if quality_analysis['has_funny_examples']:
                reference_text += "\n【有趣的评论示例】：\n"
                for i, review in enumerate(quality_analysis['funny_reviews'], 1):
                    reference_text += f"{i}. [{review['stars']}星, {review['funny']}人认为有趣] {review['text']}\n"

            # 普通评论
            reference_text += "\n【其他评论】：\n"
            for i, review in enumerate(reference_reviews[:3], 1):
                stars = review.get('stars', 'N/A')
                text = review.get('text', '')[:200]
                reference_text += f"{i}. [{stars}星] {text}\n"
        else:
            reference_text = "暂无其他用户评论。"

        # 用户最近评论示例
        recent_review_text = ""
        if user_recent_review:
            recent_review_text = f"\n你最近的一条评论示例（保持这种风格）：\n[{user_recent_review.get('stars', 'N/A')}星] {user_recent_review.get('text', '')[:300]}\n"

        prompt = f'''
你是Yelp平台上的一个真实用户，需要根据你的个人特征为一家商家写评论。

=== 你的用户资料 ===
{user_info}

{user_profile_analysis}
{recent_review_text}

=== 你要评论的商家 ===
{business_info}

=== 参考信息 ===
{reference_text}

=== 评论质量指南 ===
根据你的历史评论特征，你应该：

1. **如果你的评论通常信息价值高** (useful tendency: {user_profile_analysis}):
   - 多提供具体、实用的信息
   - 包含细节：如菜品名称、价格、服务细节、环境描述等
   - 帮助其他用户做决策

2. **如果你的评论通常比较有趣** (funny tendency: {user_profile_analysis}):
   - 可以用轻松、幽默的语气
   - 加入生动的描述或小故事
   - 但仍要保持真实性

3. **风格一致性**:
   - 评论长度：{user_profile_analysis}
   - 详细程度要匹配你的历史风格
   - 语气要自然，像你平时的风格

=== 任务要求 ===
1. **评分** (必须是1.0/2.0/3.0/4.0/5.0之一)：
   - 根据你的历史评分倾向
   - 考虑商家的实际质量

2. **评论文本** (2-4句话)：
   - 提供具体信息和细节
   - 保持与你历史风格一致
   - 真实、自然的表达

3. **输出格式** (严格遵守)：
stars: [你的评分]
review: [你的评论文本]

现在请生成你的评论：
'''
        return prompt

    def workflow(self):
        """
        主工作流程
        Returns:
            dict: {"stars": float, "review": str}
        """
        try:
            plan = self.planning(task_description=self.task)
            logging.info(f"执行计划已生成：{len(plan)}个步骤")

            # 收集信息
            user_info = None
            business_info = None

            for i, sub_task in enumerate(plan):
                logging.info(f"执行步骤 {i+1}: {sub_task['description']}")

                if 'user' in sub_task['description'].lower():
                    user_info = self.interaction_tool.get_user(
                        user_id=self.task['user_id']
                    )
                elif 'business' in sub_task['description'].lower():
                    business_info = self.interaction_tool.get_item(
                        item_id=self.task['item_id']
                    )

            # 分析用户画像（包含useful/funny/cool特征）
            reviews_user = self.interaction_tool.get_reviews(
                user_id=self.task['user_id']
            )
            user_profile_analysis = self.profile_analyzer.analyze_user_patterns(
                reviews_user
            )
            user_profile_text = self.profile_analyzer.format_user_analysis(
                user_profile_analysis
            )

            logging.info(f"用户分析完成：{user_profile_analysis['user_type']}, "
                        f"engagement_style: {user_profile_analysis['engagement_style']}")

            # 获取商家评论
            reviews_item = self.interaction_tool.get_reviews(
                item_id=self.task['item_id']
            )
            relevant_reviews = self.get_relevant_reviews(
                reviews_item,
                top_k=self.max_reference_reviews
            )

            # 分析参考评论的质量特征
            quality_analysis = self.quality_analyzer.analyze_review_qualities(
                relevant_reviews
            )
            logging.info(f"质量分析：useful示例={quality_analysis['has_useful_examples']}, "
                        f"funny示例={quality_analysis['has_funny_examples']}")

            # 使用记忆模块
            if self.use_memory and self.memory:
                for review in relevant_reviews[:3]:
                    self.memory(f"商家评论: {review.get('text', '')[:300]}")

                if reviews_user:
                    self.memory(
                        f"用户评论风格: {reviews_user[0].get('text', '')[:300]}"
                    )

            # 获取用户最近评论
            user_recent_review = reviews_user[0] if reviews_user else None

            # 构建prompt（包含质量意识）
            task_prompt = self.build_prompt(
                user_info=str(user_info),
                business_info=str(business_info),
                user_profile_analysis=user_profile_text,
                reference_reviews=relevant_reviews,
                user_recent_review=user_recent_review,
                quality_analysis=quality_analysis  # 新增
            )

            # 生成评论
            logging.info(f"开始生成评论（反思模式：{self.enable_reflection}）")
            result = self.reasoning(
                task_description=task_prompt,
                enable_reflection=self.enable_reflection
            )

            # 解析结果
            stars, review_text = self.parse_review_result(result)

            # 后处理
            if len(review_text) > 512:
                review_text = review_text[:509] + "..."
                logging.warning("评论被截断到512字符")

            logging.info(f"评论生成完成：{stars}星，长度{len(review_text)}字符")

            # 只返回stars和review（符合Track 1要求）
            return {
                "stars": stars,
                "review": review_text
            }

        except Exception as e:
            logging.error(f"Workflow错误: {e}", exc_info=True)
            return {
                "stars": 3.0,
                "review": "一般的体验。"
            }


# 为了兼容性，也导出为MySimulationAgent
MySimulationAgent = ImprovedSimulationAgent