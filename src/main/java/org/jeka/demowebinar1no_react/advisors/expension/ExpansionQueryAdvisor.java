package org.jeka.demowebinar1no_react.advisors.expension;

import lombok.Builder;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseAdvisor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.ollama.api.OllamaOptions;

import java.util.Map;

@Builder
public class ExpansionQueryAdvisor implements BaseAdvisor {

    private static final PromptTemplate template = PromptTemplate.builder()
            .template("""
            Instruction: Расширь поисковый запрос пользователя, добавив наиболее релевантные термины, связанные с меню суши-ресторана.

            СПЕЦИАЛИЗАЦИЯ:
            - Тематика: суши, роллы, маки, сеты, соусы, ингредиенты, калорийность, острые и вегетарианские позиции.
            - Атрибуты меню: цена, состав, количество штук, вес, тип (ролл/сет/маки/горячий), острота, описание.
            - Примеры ингредиентов: лосось, тунец, угорь, креветка, авокадо, огурец, сыр, соус унаги, спайси соус, тобико.
            - Примеры терминов: "острый", "вегетарианский", "с лососем", "в темпуре", "сет", "цена", "состав", "вес", "описание".

            ПРАВИЛА:
            1. Сохрани все слова из исходного вопроса.
            2. Добавь не более ПЯТИ релевантных терминов из тематики меню суши-ресторана.
            3. Не добавляй ничего, что не связано с меню, заказом или блюдами ресторана.
            4. Если вопрос не относится к еде, меню или ресторану — верни "Вопрос вне тематики меню".
            5. Результат — короткая расширенная строка, подходящая для поиска по базе меню.

            ПРИМЕРЫ:
            "что входит в сет дракон" → "что входит в сет дракон роллы угорь авокадо соус унаги"
            "цена филадельфия" → "цена ролл филадельфия лосось сливочный сыр"
            "какие есть острые роллы" → "острые роллы спайси тунец креветка"
            "вес ролла канада" → "вес ролл канада угорь соус унаги авокадо"
            "как включить свет" → "Вопрос вне тематики меню"

            Question: {question}
            Expanded query:
            """).build();


    public static final String ENRICHED_QUESTION = "ENRICHED_QUESTION";
    public static final String ORIGINAL_QUESTION = "ORIGINAL_QUESTION";
    public static final String EXPANSION_RATIO = "EXPANSION_RATIO";


    private ChatClient chatClient;

    private ChatModel chatModel;

    public static ExpansionQueryAdvisorBuilder builder(ChatModel chatModel) {
        return new ExpansionQueryAdvisorBuilder().chatClient(ChatClient.builder(chatModel)
                .defaultOptions(OllamaOptions.builder()
                        .temperature(0.0)
                        .topK(1)
                        .topP(0.1)
                        .repeatPenalty(1.0)
                        .build())
                .build());
    }

    private int order;

    @Override
    public ChatClientRequest before(ChatClientRequest chatClientRequest, AdvisorChain advisorChain) {

        String userQuestion = chatClientRequest.prompt().getUserMessage().getText();
        String enrichedQuestion = chatClient
                .prompt()
                .user(template.render(Map.of("question", userQuestion)))
                .call()
                .content();

        double ratio = enrichedQuestion.length() / (double) userQuestion.length();

        return chatClientRequest.mutate()
                .context(ORIGINAL_QUESTION, userQuestion)
                .context(ENRICHED_QUESTION, enrichedQuestion)
                .context(EXPANSION_RATIO, ratio)
                .build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse chatClientResponse, AdvisorChain advisorChain) {
        return chatClientResponse;
    }

    @Override
    public int getOrder() {
        return order;
    }
}
