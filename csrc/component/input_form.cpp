#include "tui_tool_sets.hpp"

namespace tui {
    namespace component {
        using namespace ftxui;

        InputFormBase::InputFormBase(InputFormOptions& options): InputFormOptions(options) {
            Component vertical = Container::Vertical({});
            for (int i = 0; i < options.elements_config.size(); i++) {
                components_.push_back(renderFormRow(options.elements_config[i]));
                vertical->Add(Container::Horizontal(components_[i]));
            }
            Add(vertical);
        };

        Element InputFormBase::Render() {
            std::vector<Elements> lines_;
            for (auto &row : components_) {
                Elements line;
                for (auto component : row) {
                    line.push_back(component -> Render());
                }
                lines_.push_back(line);
            }
            return gridbox(lines_);
        }

       std::vector<Component> InputFormBase::renderFormRow(ElementRowConfig row) {
            std::vector<Component> row_components;
            for (auto config : row) {
                if (config.input_type == InputType::Password) {
                    config.password = true;
                }
                Component input = Input(config);
                input = setWidth(input, default_max_input_width, default_min_input_width);
                if (config.max_input_width + config.min_input_width >= 0) {
                    input = setWidth(input, config.max_input_width, config.min_input_width);
                }
                if (config.input_type == InputType::Number) {
                    input |= CatchEvent([](Event e) {
                        return e.is_character() && !isdigit(e.character()[0]);
                    });
                }
                Component input_r = Renderer(input, [config, &default_input_style = this->default_input_style, input] {
                    return config.input_style ? config.input_style(input->Render()) : default_input_style(input->Render());
                });
                Element label = text(config.label());
                label = setWidth(label, default_max_label_width, default_min_label_width);
                if (config.max_label_width + config.min_label_width >= 0) {
                    label = setWidth(label, config.max_label_width, config.min_label_width);
                }
                Component label_r = Renderer([config, &default_label_style = this->default_label_style, label] {
                    return config.label_style ? config.label_style(label) : default_label_style(label);
                });
                row_components.push_back(label_r);
                row_components.push_back(input_r);
            }
            return row_components;
        }

        Component InputFormBase::setWidth(Component component, int max_width, int min_width) {
            return component | size(WIDTH, LESS_THAN, max_width) | size(WIDTH, GREATER_THAN, min_width);
        }
        Element InputFormBase::setWidth(Element element, int max_width, int min_width) {
            return element | size(WIDTH, LESS_THAN, max_width) | size(WIDTH, GREATER_THAN, min_width);
        }

        Component InputForm(std::vector<InputFormOptions::ElementRowConfig> elements_config, InputFormOptions options) {
            options.elements_config = std::move(elements_config);
            return std::make_shared<InputFormBase>(options);
        }
    }
}