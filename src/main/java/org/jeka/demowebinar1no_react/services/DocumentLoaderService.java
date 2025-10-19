package org.jeka.demowebinar1no_react.services;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.SneakyThrows;
import org.jeka.demowebinar1no_react.model.LoadedDocument;
import org.jeka.demowebinar1no_react.repo.ChatRepository;
import org.jeka.demowebinar1no_react.repo.DocumentRepository;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.ResourcePatternResolver;
import org.springframework.data.util.Pair;
import org.springframework.stereotype.Service;
import org.springframework.util.DigestUtils;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

@Service
public class DocumentLoaderService implements CommandLineRunner {

    @Autowired
    private DocumentRepository documentRepository;

    @Autowired
    private ResourcePatternResolver resolver;

    @Autowired
    private VectorStore vectorStore;


    @SneakyThrows
    public void loadDocuments() {
        List<Resource> resources = Arrays.stream(resolver.getResources("classpath:/knowledgebase/**/*.json")).toList();

        resources.stream()
                .map(resource -> Pair.of(resource, calcContentHash(resource)))
                .filter(pair -> !documentRepository.existsByFilenameAndContentHash(pair.getFirst().getFilename(), pair.getSecond()))
                .forEach(pair -> {
                    String json = null;
                    try {
                        json = new String(pair.getFirst().getInputStream().readAllBytes(), StandardCharsets.UTF_8);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }

                    ObjectMapper mapper = new ObjectMapper();
                    JsonNode root = null;
                    try {
                        root = mapper.readTree(json);
                    } catch (JsonProcessingException e) {
                        throw new RuntimeException(e);
                    }
                    JsonNode menuArray = root.get("menu");

                    List<Document> chunks = new ArrayList<>();

                    for (JsonNode item : menuArray) {
                        // каждый объект меню — отдельный чанк
                        String content = null;
                        try {
                            content = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(item);
                        } catch (JsonProcessingException e) {
                            throw new RuntimeException(e);
                        }
                        Document doc = new Document(content, Map.of(
                                "source", pair.getFirst().getFilename(),
                                "menu_id", item.get("id").asText(),
                                "name", item.get("name").asText(),
                                "type", item.get("type").asText()
                        ));
                        chunks.add(doc);
                    }
                    vectorStore.accept(chunks);

                    LoadedDocument loadedDocument = LoadedDocument.builder()
                            .documentType("json")
                            .chunkCount(chunks.size())
                            .filename(pair.getFirst().getFilename())
                            .contentHash(pair.getSecond())
                            .build();
                    documentRepository.save(loadedDocument);

                });


    }

    @SneakyThrows
    private String calcContentHash(Resource resource) {
        return DigestUtils.md5DigestAsHex(resource.getInputStream());
    }

    @Override
    public void run(String... args) throws Exception {
        loadDocuments();
    }
}
