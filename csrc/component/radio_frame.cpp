#include "tui_tool_sets.hpp"

namespace tui {
    namespace component {
        using namespace ftxui;
        RadioFrameBase::RadioFrameBase(RadioFrameOptions& options) : RadioFrameOptions(options) {
            content_ = Radiobox(entries, &selected()) | vscroll_indicator | frame | size(WIDTH, GREATER_THAN, min_width) | size(WIDTH, LESS_THAN, max_width) | size(HEIGHT, GREATER_THAN, min_height) | size(HEIGHT, LESS_THAN, max_height); 
            Add(content_);
        };
        Element RadioFrameBase::Render() {
            return vbox({
                hbox({
                    text(std::string(title_regx).replace(title_regx.find("%s"), 2, entries[selected()])) | bold
                }),
                separator(),
                content_ -> Render()
            });
        }

        Component RadioFrame(RadioFrameOptions options) {
            return Make<RadioFrameBase>(options);
        }

        Component RadioFrame(ConstStringListRef entries, int * selected, RadioFrameOptions options) {
            options.entries = entries;
            options.selected = selected;
            return Make<RadioFrameBase>(options);
        } 
    }
}