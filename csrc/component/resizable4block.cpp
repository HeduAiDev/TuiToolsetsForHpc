#include "tui_tool_sets.hpp"

namespace tui {
    namespace component {
        using namespace ftxui;
        Resizable4Blockbase::Resizable4Blockbase(Component block1, Component block2, Component block3, Component block4, ScreenInteractive& screen,const Resizable4BlockOptions options) 
        :   block1_(std::move(block1)),
            block2_(std::move(block2)),
            block3_(std::move(block3)),
            block4_(std::move(block4)),
            screen_(screen), 
            options_(options) {

            Add(Container::Vertical({
                Container::Horizontal({
                    block1_,
                    block2_,
                }),
                Container::Horizontal({
                    block3_,
                    block4_,
                })
            }));
        };

        Element Resizable4Blockbase::Render()  {
            int block_width = (options_.base_x() != -1  ? options_.base_x() : screen_.dimx()) / 2 + bias_x_;
            int block_height = (options_.base_y() != -1 ? options_.base_y() : screen_.dimy()) / 2 + bias_y_;
            return vbox({
                hbox({
                    (options_.placeholder_block1 && isDragging() ? options_.placeholder_block1 : block1_ -> Render()) | size(WIDTH, EQUAL, block_width),
                    getVSeparator() | reflect(vseparator_up_box_),
                    (options_.placeholder_block2 && isDragging() ? options_.placeholder_block2 : block2_ -> Render()) | flex
                }) | size(HEIGHT, EQUAL, block_height),
                getHSeparator() | reflect(hseparator_box_),
                hbox({
                    (options_.placeholder_block3 && isDragging() ? options_.placeholder_block3 : block3_ -> Render()) | size(WIDTH, EQUAL, block_width),
                    getVSeparator() | reflect(vseparator_down_box_),
                    (options_.placeholder_block4 && isDragging() ? options_.placeholder_block4 : block4_ -> Render()) | flex
                }) | size(HEIGHT, EQUAL, screen_.dimy() - block_height)
            });
        };


        bool Resizable4Blockbase::OnEvent(Event event) {
            if (event.is_mouse())
            {
                OnMouseEvent(std::move(event));
            }
            return ComponentBase::OnEvent(std::move(event));
        };

        bool Resizable4Blockbase::OnMouseEvent(Event event) {
            is_hover_hseparator_ = hseparator_box_.Contain(event.mouse().x, event.mouse().y);
            is_hover_vseparator_up_ = vseparator_up_box_.Contain(event.mouse().x, event.mouse().y);
            is_hover_vseparator_down_ = vseparator_down_box_.Contain(event.mouse().x, event.mouse().y);

            if (isDragging() && event.mouse().motion == Mouse::Released) {
                is_dragging_hseparator_ = false;
                is_dragging_vseparator_up_ = false;
                is_dragging_vseparator_down_ = false;
                return false;
            }
            if (event.mouse().button == Mouse::Left && event.mouse().motion == Mouse::Pressed && !isDragging()) {
                if (is_hover_hseparator_) {
                    is_dragging_hseparator_ = true;
                    return true;
                } else if (is_hover_vseparator_up_) {
                    is_dragging_vseparator_up_ = true;
                    return true;
                } else if (is_hover_vseparator_down_) {
                    is_dragging_vseparator_down_ = true;
                    return true;
                }
            }
            if (!isDragging()) {
                return false;
            }
            if (is_dragging_hseparator_) {
            // y direction movement
                bias_y_ += event.mouse().y - hseparator_box_.y_min;
            } else {
            // x direction movement
                bias_x_ += event.mouse().x - vseparator_up_box_.x_min;
            }
            return true;
        } 


        bool Resizable4Blockbase::isDragging() {
            return (is_dragging_hseparator_ || is_dragging_vseparator_up_ || is_dragging_vseparator_down_ );
        };
        
        Element Resizable4Blockbase::getVSeparator() {
            return (is_hover_vseparator_up_ || is_hover_vseparator_down_) ? options_.separator_hover_func() : options_.separator_func();
        };

        Element Resizable4Blockbase::getHSeparator() {
            return (is_hover_hseparator_) ? options_.separator_hover_func() : options_.separator_func();
        };

        Component Resizable4Block(Component block1, Component block2, Component block3, Component block4, ScreenInteractive& screen, Resizable4BlockOptions options) {
            return Make<Resizable4Blockbase>(std::move(block1), std::move(block2), std::move(block3), std::move(block4), screen, std::move(options));
        }
    }
}